"""
抽象引擎：将多个相关 Episodic 记忆聚合成 LTM 结论

工作流程:
1. 发现主题相似的 M3 片段
2. 使用 LLM 提炼抽象结论
3. 存入 LTM (Neo4j)
4. 建立 M3 → LTM 关系
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from repositories.mongodb_repository import MongoDBRepository
from repositories.neo4j_repository import Neo4jRepository
from schemas.graph import LTMNode
from services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)


@dataclass
class AbstractionResult:
    """抽象结果"""

    topic: str
    conclusion: str
    conditions: List[str]
    confidence: float
    source_chunk_ids: List[str]
    ltm_topic_version: str


class AbstractionEngine:
    """抽象引擎"""

    MIN_EPISODES = 5
    MIN_AVG_IMPORTANCE = 0.6
    SIMILARITY_THRESHOLD = 0.75
    MAX_TOPICS_PER_RUN = 10

    def __init__(
        self,
        mongo_repo: MongoDBRepository = None,
        neo4j_repo: Neo4jRepository = None,
        feedback_service: FeedbackService = None,
        llm_model: Any = None,
    ):
        self.mongo_repo = mongo_repo or MongoDBRepository()
        self.neo4j_repo = neo4j_repo or Neo4jRepository()
        self.feedback_service = feedback_service or FeedbackService(self.mongo_repo)
        self.llm_model = llm_model

    def find_related_episodes(
        self,
        topic: str,
        min_count: int = None,
        min_avg_importance: float = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        查找与主题相关的 episodic 记忆

        Args:
            topic: 查询主题
            min_count: 最少片段数
            min_avg_importance: 最低平均重要性

        Returns:
            (相关片段列表, 平均重要性)
        """
        min_count = min_count or self.MIN_EPISODES
        min_avg_importance = min_avg_importance or self.MIN_AVG_IMPORTANCE

        chunks = self.mongo_repo.get_chunks_by_memory_type("episodic")

        related = []
        for chunk in chunks:
            text = chunk.get("text", "")
            importance = chunk.get("importance_score", 0.0)

            if self._text_matches_topic(text, topic):
                related.append(chunk)

        if not related:
            return [], 0.0

        avg_importance = sum(c.get("importance_score", 0) for c in related) / len(
            related
        )

        return related, avg_importance

    def _text_matches_topic(self, text: str, topic: str) -> bool:
        """简单判断文本是否匹配主题"""
        topic_lower = topic.lower()
        text_lower = text.lower()
        return topic_lower in text_lower or any(
            word in text_lower for word in topic_lower.split()
        )

    def can_abstraction_trigger(self, topic: str) -> bool:
        """检查是否满足抽象触发条件"""
        episodes, avg_importance = self.find_related_episodes(topic)
        return (
            len(episodes) >= self.MIN_EPISODES
            and avg_importance >= self.MIN_AVG_IMPORTANCE
        )

    def abstract_to_ltm(
        self,
        topic: str,
        source_chunk_ids: List[str] = None,
    ) -> Optional[AbstractionResult]:
        """
        将相关片段抽象为 LTM 结论

        Args:
            topic: 主题
            source_chunk_ids: 可选，指定源片段ID

        Returns:
            抽象结果或None（如果不满足条件）
        """
        if source_chunk_ids:
            episodes = [self.mongo_repo.get_chunk(cid) for cid in source_chunk_ids]
            episodes = [e for e in episodes if e]
            avg_importance = (
                sum(e.get("importance_score", 0) for e in episodes) / len(episodes)
                if episodes
                else 0
            )
        else:
            episodes, avg_importance = self.find_related_episodes(topic)

        if len(episodes) < self.MIN_EPISODES:
            logger.info(
                f"Not enough episodes for abstraction: {len(episodes)} < {self.MIN_EPISODES}"
            )
            return None

        if avg_importance < self.MIN_AVG_IMPORTANCE:
            logger.info(
                f"Average importance too low: {avg_importance:.2f} < {self.MIN_AVG_IMPORTANCE:.2f}"
            )
            return None

        fragment_texts = [e.get("text", "") for e in episodes[:10]]
        fragment_text = "\n\n---\n\n".join(fragment_texts)

        if self.llm_model:
            conclusion, conditions, confidence = self._llm_abstract(
                fragment_text, topic
            )
        else:
            conclusion, conditions, confidence = self._rule_based_abstract(
                fragment_text, topic
            )

        chunk_ids = [e.get("chunk_id") for e in episodes]

        ltm = LTMNode(
            topic=topic,
            version=self._get_next_version(topic),
            conclusion=conclusion,
            conditions=conditions,
            confidence=confidence,
            sources=["abstracted"],
            source_chunk_ids=chunk_ids,
        )

        topic_version = self.neo4j_repo.save_ltm(ltm)

        for chunk_id in chunk_ids:
            self.neo4j_repo.link_episodic_to_ltm(chunk_id, topic_version)

        logger.info(f"Created LTM {topic_version} from {len(episodes)} episodes")

        return AbstractionResult(
            topic=topic,
            conclusion=conclusion,
            conditions=conditions,
            confidence=confidence,
            source_chunk_ids=chunk_ids,
            ltm_topic_version=topic_version,
        )

    def _get_next_version(self, topic: str) -> int:
        """获取下一个版本号"""
        existing = self.neo4j_repo.get_ltm_by_topic(topic)
        if not existing:
            return 1
        return max(ltm.get("version", 1) for ltm in existing) + 1

    def _llm_abstract(
        self,
        fragments: str,
        topic: str,
    ) -> Tuple[str, List[str], float]:
        """使用 LLM 进行抽象"""
        prompt = f"""
You are a cognitive memory abstraction system. Given the following related memory fragments about "{topic}", extract a general principle/conclusion.

Memory Fragments:
{fragments[:3000]}

Output format (JSON):
{{
  "conclusion": "A concise general principle (1-2 sentences)",
  "conditions": ["condition1", "condition2"],
  "confidence": 0.0-1.0
}}

Rules:
- Conclusion should be generalizable beyond these specific fragments
- Conditions specify when this conclusion applies
- Confidence reflects certainty based on evidence quality
"""
        try:
            response = self.llm_model.chat(prompt, temperature=0.3)
            import json

            data = json.loads(response)
            return (
                data.get("conclusion", ""),
                data.get("conditions", []),
                float(data.get("confidence", 0.5)),
            )
        except Exception as e:
            logger.error(f"LLM abstraction failed: {e}")
            return self._rule_based_abstract(fragments, topic)

    def _rule_based_abstract(
        self,
        fragments: str,
        topic: str,
    ) -> Tuple[str, List[str], float]:
        """基于规则的抽象（LLM不可用时的降级方案）"""
        first_chunk = fragments.split("\n")[0][:500]
        conclusion = f"Summary of {topic}: {first_chunk}"
        conditions = ["general case"]
        confidence = 0.5
        return conclusion, conditions, confidence

    def auto_discover_and_abstract(self) -> List[AbstractionResult]:
        """
        自动发现可抽象的主题并执行抽象

        Returns:
            抽象结果列表
        """
        results = []

        chunks = self.mongo_repo.get_chunks_by_memory_type("episodic")
        topics = self._discover_topics(chunks)

        for topic in topics[: self.MAX_TOPICS_PER_RUN]:
            if self.can_abstraction_trigger(topic):
                result = self.abstract_to_ltm(topic)
                if result:
                    results.append(result)

        logger.info(f"Auto-abstract completed: {len(results)} LTM nodes created")
        return results

    def _discover_topics(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """从chunks中发现可能的主题"""
        topics = set()
        for chunk in chunks:
            text = chunk.get("text", "")
            importance = chunk.get("importance_score", 0)

            if importance > 0.5:
                words = text.split()[:50]
                significant = [w for w in words if len(w) > 5 and w[0].isupper()]
                for word in significant[:3]:
                    topics.add(word)

        return list(topics)

    def get_abstraction_candidates(self) -> List[Dict[str, Any]]:
        """获取所有满足抽象条件的候选主题"""
        candidates = []
        chunks = self.mongo_repo.get_chunks_by_memory_type("episodic")

        topics = self._discover_topics(chunks)
        for topic in topics:
            episodes, avg_importance = self.find_related_episodes(topic)
            if (
                len(episodes) >= self.MIN_EPISODES
                and avg_importance >= self.MIN_AVG_IMPORTANCE
            ):
                candidates.append(
                    {
                        "topic": topic,
                        "episode_count": len(episodes),
                        "avg_importance": avg_importance,
                    }
                )

        return sorted(candidates, key=lambda x: x["avg_importance"], reverse=True)

    def close(self):
        """关闭数据库连接"""
        self.mongo_repo.close()
        self.neo4j_repo.close()


class AbstractionConfig:
    """抽象引擎配置"""

    MIN_EPISODES = 5
    MIN_AVG_IMPORTANCE = 0.6
    SIMILARITY_THRESHOLD = 0.75
    MAX_TOPICS_PER_RUN = 10

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "min_episodes": cls.MIN_EPISODES,
            "min_avg_importance": cls.MIN_AVG_IMPORTANCE,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_topics_per_run": cls.MAX_TOPICS_PER_RUN,
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    engine = AbstractionEngine()
    candidates = engine.get_abstraction_candidates()
    print(f"Found {len(candidates)} abstraction candidates")
    for c in candidates[:5]:
        print(
            f"  - {c['topic']}: {c['episode_count']} episodes, avg importance: {c['avg_importance']:.2f}"
        )
    engine.close()
