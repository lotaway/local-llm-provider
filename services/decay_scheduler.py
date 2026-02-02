"""
衰减调度器：定期执行记忆重要性衰减

衰减公式：
    importance(t) = importance_0 * exp(-λ * Δt)

复习强化：
    importance = importance * (1 + review_count * k)
"""

import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from constants import LEARNING
from repositories.mongodb_repository import MongoDBRepository
from services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)


class DecayScheduler:
    DEFAULT_DECAY_RATE = 0.01
    DEFAULT_REINFORCEMENT_FACTOR = 0.05
    IMPORTANCE_THRESHOLD = 0.1
    STALE_DAYS = 30

    def __init__(
        self,
        mongo_repo: MongoDBRepository = None,
        feedback_service: FeedbackService = None,
        decay_rate: float = None,
        reinforcement_factor: float = None,
    ):
        self.mongo_repo = mongo_repo or MongoDBRepository()
        self.feedback_service = feedback_service or FeedbackService(self.mongo_repo)
        self.decay_rate = decay_rate or self.DEFAULT_DECAY_RATE
        self.reinforcement_factor = (
            reinforcement_factor or self.DEFAULT_REINFORCEMENT_FACTOR
        )

    def calculate_decay(self, chunk: Dict[str, Any]) -> float:
        """
        计算chunk的衰减后重要性

        Args:
            chunk: chunk数据字典

        Returns:
            衰减后的重要性评分
        """
        current_score = chunk.get("importance_score", 0.5)
        last_reviewed = chunk.get("last_reviewed")
        decay_rate = chunk.get("decay_rate", self.decay_rate)
        review_count = chunk.get("review_count", 0)

        if last_reviewed is None:
            last_reviewed = chunk.get("created_at", datetime.utcnow())

        if isinstance(last_reviewed, str):
            last_reviewed = datetime.fromisoformat(last_reviewed.replace("Z", "+00:00"))

        delta_days = (datetime.utcnow() - last_reviewed).total_seconds() / 86400

        decayed_score = current_score * math.exp(-decay_rate * delta_days)

        reinforcement = 1 + (review_count * self.reinforcement_factor)
        final_score = decayed_score * reinforcement

        return max(0.0, min(1.0, final_score))

    def apply_decay_to_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        对单个chunk应用衰减

        Args:
            chunk_id: chunk ID

        Returns:
            更新后的chunk状态
        """
        chunk = self.mongo_repo.get_chunk(chunk_id)
        if not chunk:
            logger.warning(f"Chunk {chunk_id} not found")
            return {}

        new_score = self.calculate_decay(chunk)
        self.mongo_repo.update_importance_score(chunk_id, new_score)

        logger.debug(
            f"Applied decay to {chunk_id}: {chunk['importance_score']:.4f} -> {new_score:.4f}"
        )

        return self.mongo_repo.get_chunk(chunk_id)

    def apply_decay_all(self, memory_type: str = None) -> Dict[str, Any]:
        """
        对所有chunk应用衰减

        Args:
            memory_type: 可选，只处理特定类型

        Returns:
            处理统计
        """
        if not LEARNING:
            logger.info("LEARNING=false, skipping decay application")
            return {"skipped": True, "reason": "LEARNING disabled"}

        chunks = (
            self.mongo_repo.get_chunks_by_memory_type(memory_type)
            if memory_type
            else list(self.mongo_repo.chunks.find({}))
        )

        updated = 0
        degraded = 0
        reinforced = 0
        errors = 0

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            try:
                old_score = chunk.get("importance_score", 0.5)
                new_score = self.calculate_decay(chunk)

                self.mongo_repo.update_importance_score(chunk_id, new_score)
                updated += 1

                if new_score < old_score:
                    degraded += 1
                else:
                    reinforced += 1

            except Exception as e:
                logger.error(f"Error applying decay to {chunk_id}: {e}")
                errors += 1

        stats = {
            "total": len(chunks),
            "updated": updated,
            "degraded": degraded,
            "reinforced": reinforced,
            "errors": errors,
        }

        logger.info(f"Decay applied: {stats}")
        return stats

    def cleanup_low_importance(self, threshold: float = None) -> List[str]:
        """
        清理低重要性记忆

        Args:
            threshold: 低于此重要性的chunk将被标记为待清理

        Returns:
            被清理的chunk_id列表
        """
        if not LEARNING:
            logger.info("LEARNING=false, skipping cleanup")
            return []

        threshold = threshold or self.IMPORTANCE_THRESHOLD
        stale_chunks = self.mongo_repo.get_low_importance_chunks(
            threshold=threshold, limit=1000
        )

        cleaned = []
        for chunk in stale_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                self.mongo_repo.chunks.update_one(
                    {"chunk_id": chunk_id}, {"$set": {"memory_type": "discarded"}}
                )
                cleaned.append(chunk_id)

        if cleaned:
            logger.info(f"Marked {len(cleaned)} low-importance chunks as discarded")

        return cleaned

    def get_decay_stats(self) -> Dict[str, Any]:
        """
        获取衰减统计信息
        """
        total_chunks = self.mongo_repo.chunks.count_documents({})
        episodic_chunks = self.mongo_repo.chunks.count_documents(
            {"memory_type": "episodic"}
        )
        discarded_chunks = self.mongo_repo.chunks.count_documents(
            {"memory_type": "discarded"}
        )

        avg_importance = self.mongo_repo.chunks.aggregate(
            [{"$group": {"_id": None, "avg": {"$avg": "$importance_score"}}}]
        )

        avg_val = list(avg_importance)
        avg_score = avg_val[0]["avg"] if avg_val else 0.0

        return {
            "total_chunks": total_chunks,
            "episodic_chunks": episodic_chunks,
            "discarded_chunks": discarded_chunks,
            "average_importance": round(avg_score, 4),
            "decay_rate": self.decay_rate,
            "reinforcement_factor": self.reinforcement_factor,
        }

    def close(self):
        """关闭数据库连接"""
        self.feedback_service.close()


class DecayConfig:
    """衰减配置"""

    DECAY_RATE = 0.01
    REINFORCEMENT_FACTOR = 0.05
    IMPORTANCE_THRESHOLD = 0.1
    STALE_DAYS = 30
    BATCH_SIZE = 100

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "decay_rate": cls.DECAY_RATE,
            "reinforcement_factor": cls.REINFORCEMENT_FACTOR,
            "importance_threshold": cls.IMPORTANCE_THRESHOLD,
            "stale_days": cls.STALE_DAYS,
            "batch_size": cls.BATCH_SIZE,
        }


def run_decay_job():
    """
    独立运行的衰减任务（供调度器调用）
    """
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    scheduler = DecayScheduler()
    stats = scheduler.apply_decay_all()
    scheduler.cleanup_low_importance()
    scheduler.close()

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = run_decay_job()
    print(f"Decay job completed: {stats}")
