"""
版本控制管理器：处理知识演进中的版本和冲突

功能：
1. 冲突检测
2. 版本策略（合并/保留）
3. 版本链管理
4. 来源追溯
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from constants import LEARNING
from repositories.neo4j_repository import Neo4jRepository
from repositories.mongodb_repository import MongoDBRepository
from schemas.graph import LTMNode

logger = logging.getLogger(__name__)


@dataclass
class ConflictResult:
    """冲突检测结果"""

    has_conflict: bool
    conflicting_versions: List[Dict[str, Any]]
    similarity_score: float
    recommended_action: str  # "merge" | "create_version" | "skip"


class VersionManager:
    """版本控制管理器"""

    CONFLICT_SIMILARITY_THRESHOLD = 0.7
    MERGE_SIMILARITY_THRESHOLD = 0.9

    def __init__(
        self,
        neo4j_repo: Neo4jRepository = None,
        mongo_repo: MongoDBRepository = None,
    ):
        self.neo4j_repo = neo4j_repo or Neo4jRepository()
        self.mongo_repo = mongo_repo or MongoDBRepository()

    def detect_conflict(self, topic: str, new_conclusion: str) -> ConflictResult:
        """
        检测新结论是否与现有LTM冲突

        Args:
            topic: 主题
            new_conclusion: 新结论

        Returns:
            冲突检测结果
        """
        existing = self.neo4j_repo.get_ltm_by_topic(topic)
        if not existing:
            return ConflictResult(
                has_conflict=False,
                conflicting_versions=[],
                similarity_score=0.0,
                recommended_action="create",
            )

        conflicting = []
        max_similarity = 0.0

        for lt in existing:
            similarity = self._text_similarity(new_conclusion, lt.get("conclusion", ""))
            max_similarity = max(max_similarity, similarity)

            if similarity < self.CONFLICT_SIMILARITY_THRESHOLD:
                conflicting.append(lt)

        has_conflict = (
            max_similarity < self.CONFLICT_SIMILARITY_THRESHOLD and len(existing) > 0
        )

        if has_conflict:
            if max_similarity > self.MERGE_SIMILARITY_THRESHOLD:
                action = "merge"
            else:
                action = "create_version"
        else:
            action = "update" if max_similarity > 0.5 else "create"

        return ConflictResult(
            has_conflict=has_conflict,
            conflicting_versions=conflicting,
            similarity_score=max_similarity,
            recommended_action=action,
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单版，可替换为语义相似度）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def upsert_ltm(self, ltm: LTMNode) -> str:
        """
        智能保存LTM，处理版本和冲突

        Args:
            ltm: LTM节点

        Returns:
            保存的topic_version
        """
        if not LEARNING:
            logger.info("LEARNING=false, skipping LTM upsert")
            return ""

        conflict = self.detect_conflict(ltm.topic, ltm.conclusion)

        if conflict.recommended_action == "merge":
            return self._merge_and_update(ltm, conflict.conflicting_versions[0])
        elif conflict.recommended_action == "create_version":
            return self._create_version(ltm, conflict.conflicting_versions)
        elif conflict.recommended_action == "update":
            return self._update_existing(ltm, conflict.conflicting_versions[0])
        else:
            return self._create_new(ltm)

    def _merge_and_update(self, new_ltm: LTMNode, existing: Dict[str, Any]) -> str:
        """合并相同源的结论"""
        new_version = existing.get("version", 1) + 1

        merged_sources = list(set(existing.get("sources", []) + new_ltm.sources))

        merged_chunk_ids = list(
            set(existing.get("source_chunk_ids", []) + new_ltm.source_chunk_ids)
        )

        merged_ltm = LTMNode(
            topic=new_ltm.topic,
            version=new_version,
            conclusion=new_ltm.conclusion,
            conditions=new_ltm.conditions or existing.get("conditions", []),
            confidence=max(new_ltm.confidence, existing.get("confidence", 0.5)),
            sources=merged_sources,
            source_chunk_ids=merged_chunk_ids,
        )

        topic_version = self.neo4j_repo.save_ltm(merged_ltm)

        self._link_sources_to_ltm(merged_ltm.source_chunk_ids, topic_version)

        logger.info(f"Merged LTM: {topic_version}")
        return topic_version

    def _create_version(self, new_ltm: LTMNode, existing: List[Dict[str, Any]]) -> str:
        """创建新版本（冲突时保留旧版本）"""
        new_ltm.version = self._get_next_version(new_ltm.topic)
        topic_version = self.neo4j_repo.save_ltm(new_ltm)

        for lt in existing[:1]:
            old_version = lt.get("topic_version")
            if old_version:
                self.neo4j_repo.create_ltm_version_relation(old_version, topic_version)

        self._link_sources_to_ltm(new_ltm.source_chunk_ids, topic_version)

        logger.info(f"Created new version: {topic_version}")
        return topic_version

    def _update_existing(self, new_ltm: LTMNode, existing: Dict[str, Any]) -> str:
        """更新现有版本"""
        new_ltm.version = existing.get("version", 1)
        new_ltm.sources = list(set(existing.get("sources", []) + new_ltm.sources))
        new_ltm.source_chunk_ids = list(
            set(existing.get("source_chunk_ids", []) + new_ltm.source_chunk_ids)
        )
        return self.neo4j_repo.save_ltm(new_ltm)

    def _create_new(self, ltm: LTMNode) -> str:
        """创建全新的LTM"""
        ltm.version = 1
        topic_version = self.neo4j_repo.save_ltm(ltm)
        self._link_sources_to_ltm(ltm.source_chunk_ids, topic_version)
        logger.info(f"Created new LTM: {topic_version}")
        return topic_version

    def _get_next_version(self, topic: str) -> int:
        """获取下一个版本号"""
        existing = self.neo4j_repo.get_ltm_by_topic(topic)
        if not existing:
            return 1
        return max(ltm.get("version", 1) for ltm in existing) + 1

    def _link_sources_to_ltm(self, chunk_ids: List[str], topic_version: str):
        """将源chunk链接到LTM"""
        for chunk_id in chunk_ids:
            try:
                self.neo4j_repo.link_episodic_to_ltm(chunk_id, topic_version)
            except Exception as e:
                logger.warning(f"Failed to link {chunk_id} to {topic_version}: {e}")

    def get_version_chain(self, topic: str) -> List[Dict[str, Any]]:
        """获取版本链"""
        return self.neo4j_repo.get_ltm_versions_chain(topic)

    def get_version_history(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取版本历史"""
        chain = self.get_version_chain(topic)
        return chain[:limit]

    def compare_versions(
        self, topic: str, version1: int, version2: int
    ) -> Dict[str, Any]:
        """比较两个版本"""
        v1 = self.neo4j_repo.get_ltm_by_topic_version(topic, version1)
        v2 = self.neo4j_repo.get_ltm_by_topic_version(topic, version2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        similarity = self._text_similarity(
            v1.get("conclusion", ""), v2.get("conclusion", "")
        )

        return {
            "v1": {
                "version": v1.get("version"),
                "conclusion": v1.get("conclusion")[:200],
                "confidence": v1.get("confidence"),
                "created_at": v1.get("created_at"),
            },
            "v2": {
                "version": v2.get("version"),
                "conclusion": v2.get("conclusion")[:200],
                "confidence": v2.get("confidence"),
                "created_at": v2.get("created_at"),
            },
            "similarity": similarity,
            "significant_change": similarity < 0.7,
        }

    def rollback_to_version(self, topic: str, version: int) -> bool:
        """回滚到指定版本"""
        if not LEARNING_ENABLED:
            logger.info("LEARNING=false, skipping rollback")
            return False

        target = self.neo4j_repo.get_ltm_by_topic_version(topic, version)
        if not target:
            logger.warning(f"Version {version} not found for topic {topic}")
            return False

        new_ltm = LTMNode(
            topic=topic,
            version=self._get_next_version(topic) + 1,
            conclusion=target.get("conclusion", ""),
            conditions=target.get("conditions", []),
            confidence=target.get("confidence", 0.5),
            sources=target.get("sources", []) + ["rollback"],
            source_chunk_ids=target.get("source_chunk_ids", []),
        )

        new_version = self.neo4j_repo.save_ltm(new_ltm)

        latest = self.neo4j_repo.get_latest_ltm(topic)
        if latest:
            self.neo4j_repo.create_ltm_version_relation(
                latest.get("topic_version"), new_version
            )

        logger.info(
            f"Rolled back to version {version}, created new version {new_ltm.version}"
        )
        return True

    def get_conflict_report(self, topic: str) -> Dict[str, Any]:
        """生成冲突报告"""
        existing = self.neo4j_repo.get_ltm_by_topic(topic)

        if len(existing) <= 1:
            return {
                "topic": topic,
                "has_conflicts": False,
                "version_count": len(existing),
                "message": "No conflicts detected",
            }

        conflicts = []
        for i, lt1 in enumerate(existing):
            for lt2 in existing[i + 1 :]:
                similarity = self._text_similarity(
                    lt1.get("conclusion", ""), lt2.get("conclusion", "")
                )
                if similarity < self.CONFLICT_SIMILARITY_THRESHOLD:
                    conflicts.append(
                        {
                            "v1": lt1.get("version"),
                            "v2": lt2.get("version"),
                            "similarity": similarity,
                            "conflict": True,
                        }
                    )

        return {
            "topic": topic,
            "has_conflicts": len(conflicts) > 0,
            "version_count": len(existing),
            "conflicts": conflicts,
            "recommendation": (
                "Consider merging similar versions" if conflicts else "OK"
            ),
        }

    def close(self):
        """关闭连接"""
        self.neo4j_repo.close()
        self.mongo_repo.close()


class VersionConfig:
    """版本控制配置"""

    CONFLICT_SIMILARITY_THRESHOLD = 0.7
    MERGE_SIMILARITY_THRESHOLD = 0.9

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "conflict_similarity_threshold": cls.CONFLICT_SIMILARITY_THRESHOLD,
            "merge_similarity_threshold": cls.MERGE_SIMILARITY_THRESHOLD,
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    vm = VersionManager()
    report = vm.get_conflict_report("test_topic")
    print(f"Conflict Report: {report}")
    vm.close()
