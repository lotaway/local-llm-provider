from typing import Any, Dict, List, Optional
import logging

from constants import LEARNING
from interfaces.memory_types import IMemoryRepository
from interfaces.abstraction_types import IAbstractionEngine, ITopicExtractor
from interfaces.ltm_index_types import ILtmIndexRepository
from schemas.graph import LTMNode
from services.topic_extractor import GraphKeywordTopicExtractor

logger = logging.getLogger(__name__)


class AbstractionEngine(IAbstractionEngine):
    MIN_EPISODES = 5
    MIN_AVG_IMPORTANCE = 0.6

    def __init__(
        self,
        mongo_repo: IMemoryRepository = None,
        neo4j_repo=None,
        feedback_service=None,
        llm=None,
        topic_extractor: ITopicExtractor = None,
        ltm_index: ILtmIndexRepository = None,
    ):
        self._repo = mongo_repo
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
                except Exception as error:
                    logger.warning(
                        "Failed to scan alternative episodic memories",
                        extra={"topic": topic, "error": str(error)},
                    )
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

        llm_conclusion = self._summarize_with_llm(topic, sources)
        if llm_conclusion:
            conclusion = llm_conclusion

        ltm = LTMNode(
            topic=topic,
            conclusion=conclusion,
            conditions=conditions,
            confidence=min(0.9, 0.5 + avg_importance * 0.4),
            sources=sources,
            source_chunk_ids=source_chunk_ids,
        )

        ltm_dict = ltm.model_dump() if hasattr(ltm, "model_dump") else ltm.dict()

        self._persist_ltm(ltm, ltm_dict, sources, source_chunk_ids)

        return {"topic": topic, "episodes_count": len(episodes), "avg_importance": avg_importance, "ltm": ltm_dict}

    def _summarize_with_llm(self, topic: str, sources: List[str]) -> Optional[str]:
        if not self._llm:
            return None
        try:
            prompt = f"Summarize a standard conclusion for topic '{topic}' from episodes: {sources[:5]}. Return JSON with conclusion, conditions, confidence."
            resp = None
            if hasattr(self._llm, "chat_at_once"):
                resp = self._llm.chat_at_once(prompt, temperature=0.2)
            elif hasattr(self._llm, "invoke"):
                resp = self._llm.invoke(prompt)
            if resp:
                return str(resp)[:500]
        except Exception as error:
            logger.warning("LLM abstraction summarization failed", extra={"topic": topic, "error": str(error)})
        return None

    def _persist_ltm(
        self,
        ltm: Any,
        ltm_dict: Dict[str, Any],
        sources: List[str],
        source_chunk_ids: List[str],
    ) -> None:
        if self._neo4j_repo and hasattr(self._neo4j_repo, "save_ltm"):
            try:
                self._neo4j_repo.save_ltm(ltm)
            except Exception as error:
                logger.error(
                    "Failed to persist abstracted LTM node",
                    extra={"topic": ltm.topic, "error": str(error)},
                )
        if self._ltm_index is not None:
            try:
                self._ltm_index.save_ltm_vector(ltm_dict)
                self._ltm_index.index_ltm_keywords(ltm_dict)
            except Exception as error:
                logger.error(
                    "Failed to index abstracted LTM vector",
                    extra={"topic": str(ltm.topic), "error": str(error)},
                )

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
            except Exception as error:
                logger.warning("Failed to load episodic memories", extra={"topic": topic, "error": str(error)})
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
