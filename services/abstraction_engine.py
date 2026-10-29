from typing import List, Dict, Optional
from constants import LEARNING
from interfaces.memory_types import IMemoryRepository
from interfaces.abstraction_types import IAbstractionEngine, ITopicExtractor
from interfaces.ltm_index_types import ILtmIndexRepository
from schemas.graph import LTMNode
from services.topic_extractor import GraphKeywordTopicExtractor


class AbstractionEngine(IAbstractionEngine):
    MIN_EPISODES = 5
    MIN_AVG_IMPORTANCE = 0.6

    def __init__(
        self,
        memory_repo: IMemoryRepository = None,
        mongo_repo=None,
        neo4j_repo=None,
        feedback_service=None,
        llm=None,
        topic_extractor: ITopicExtractor = None,
        ltm_index: ILtmIndexRepository = None,
    ):
        self._repo = memory_repo or mongo_repo
        self._neo4j_repo = neo4j_repo
        self._feedback_service = feedback_service
        self._llm = llm
        self._topic_extractor = topic_extractor or GraphKeywordTopicExtractor(neo4j_repo)
        self._ltm_index = ltm_index

    def can_abstraction_trigger(self, topic: str) -> bool:
        episodes, avg_importance = self._find_related(topic)
        if not episodes:
            # try alternative
            if self._repo and hasattr(self._repo, "get_chunks_by_memory_type"):
                try:
                    all_eps = self._repo.get_chunks_by_memory_type("episodic") or []
                    episodes = [e for e in all_eps if topic.lower() in str(e.get("text") or e.get("content") or "").lower()]
                    avg_importance = sum(e.get("importance_score", e.get("importance", 0)) for e in episodes) / len(episodes) if episodes else 0
                except Exception:
                    pass
        return len(episodes) >= self.MIN_EPISODES and avg_importance >= self.MIN_AVG_IMPORTANCE

    def abstract_to_ltm(self, topic: str) -> Optional[Dict]:
        if not LEARNING:
            return None

        episodes, avg_importance = self._find_related(topic)
        if not self._can_trigger(len(episodes), avg_importance):
            return None

        conclusion = f"Abstracted conclusion for {topic} based on {len(episodes)} episodes."
        conditions = []
        sources = []
        source_chunk_ids = []
        for e in episodes[:10]:
            if e.get("text"):
                sources.append(str(e.get("text"))[:200])
            cid = e.get("chunk_id") or e.get("id")
            if cid:
                source_chunk_ids.append(str(cid))

        try:
            if self._llm:
                prompt = f"Summarize a standard conclusion for topic '{topic}' from episodes: {sources[:5]}. Return JSON with conclusion, conditions, confidence."
                resp = None
                if hasattr(self._llm, "chat_at_once"):
                    resp = self._llm.chat_at_once(prompt, temperature=0.2)
                elif hasattr(self._llm, "invoke"):
                    resp = self._llm.invoke(prompt)
                if resp:
                    # simple extraction
                    conclusion = resp[:500]
        except Exception:
            pass

        ltm = LTMNode(
            topic=topic,
            conclusion=conclusion,
            conditions=conditions,
            confidence=min(0.9, 0.5 + avg_importance * 0.4),
            sources=sources,
            source_chunk_ids=source_chunk_ids,
        )

        ltm_dict = ltm.model_dump() if hasattr(ltm, "model_dump") else ltm.dict()

        # write to neo4j
        if self._neo4j_repo and hasattr(self._neo4j_repo, "save_ltm"):
            try:
                self._neo4j_repo.save_ltm(ltm)
            except Exception:
                pass

        if self._ltm_index is not None:
            self._ltm_index.save_ltm_vector(ltm_dict)
            self._ltm_index.index_ltm_keywords(ltm_dict)

        return {"topic": topic, "episodes_count": len(episodes), "avg_importance": avg_importance, "ltm": ltm_dict}

    def auto_discover_and_abstract(self) -> List[Dict]:
        if not LEARNING:
            return []

        topics = self._topic_extractor.extract(self._episodic_memories())
        return [
            result
            for topic in topics
            if self.can_abstraction_trigger(topic)
            for result in [self.abstract_to_ltm(topic)]
            if result
        ]

    def _episodic_memories(self) -> List[Dict]:
        if self._repo and hasattr(self._repo, "get_by_type"):
            return self._repo.get_by_type("episodic")
        if self._repo and hasattr(self._repo, "get_chunks_by_memory_type"):
            return self._repo.get_chunks_by_memory_type("episodic") or []
        return []

    def _find_related(self, topic: str) -> tuple:
        episodes = []
        if self._repo and hasattr(self._repo, "get_by_type"):
            episodes = [e for e in self._repo.get_by_type("episodic") if topic.lower() in str(e.get("text") or e.get("content") or "").lower()]
        elif self._repo and hasattr(self._repo, "get_chunks_by_memory_type"):
            try:
                all_eps = self._repo.get_chunks_by_memory_type("episodic") or []
                episodes = [e for e in all_eps if topic.lower() in str(e.get("text") or e.get("content") or "").lower()]
            except Exception:
                episodes = []
        avg = sum(e.get("importance_score", e.get("importance", 0)) for e in episodes) / len(episodes) if episodes else 0
        return episodes, avg

    def _can_trigger(self, count: int, avg_importance: float) -> bool:
        return count >= self.MIN_EPISODES and avg_importance >= self.MIN_AVG_IMPORTANCE

    # compatibility
    def find_related_episodes(self, topic: str):
        eps, avg = self._find_related(topic)
        return eps, avg

    def get_abstraction_candidates(self):
        return self.auto_discover_and_abstract()
