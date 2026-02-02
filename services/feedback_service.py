"""
反馈服务：处理记忆使用反馈和重要性评分更新

反馈类型：
- on_recall: 记忆被检索召回时触发
- on_usage: 记忆被实际使用（参与任务成功）时触发
- on_task_success: 整体任务成功，多个相关记忆一起强化
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from repositories.mongodb_repository import MongoDBRepository

logger = logging.getLogger(__name__)


class FeedbackService:
    DEFAULT_RECALL_DELTA = 0.05
    DEFAULT_USAGE_DELTA = 0.15
    MAX_IMPORTANCE = 1.0
    MIN_IMPORTANCE = 0.0

    def __init__(self, mongo_repo: MongoDBRepository = None):
        self.mongo_repo = mongo_repo or MongoDBRepository()

    def on_recall(self, chunk_id: str, delta: float = None) -> Dict[str, Any]:
        """
        记忆被召回时的反馈

        Args:
            chunk_id: 被召回的记忆chunk ID
            delta: 重要性增量，默认0.05

        Returns:
            更新后的记忆状态
        """
        delta = delta or self.DEFAULT_RECALL_DELTA
        return self._increment_importance(chunk_id, delta)

    def on_recall_batch(
        self, chunk_ids: List[str], delta: float = None
    ) -> List[Dict[str, Any]]:
        """
        批量记忆召回反馈

        Args:
            chunk_ids: 被召回的记忆ID列表
            delta: 重要性增量

        Returns:
            更新后的记忆状态列表
        """
        delta = delta or self.DEFAULT_RECALL_DELTA
        results = []
        for chunk_id in chunk_ids:
            try:
                result = self.on_recall(chunk_id, delta)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to update recall feedback for {chunk_id}: {e}")
                results.append({"chunk_id": chunk_id, "error": str(e)})
        return results

    def on_usage(
        self, chunk_id: str, success: bool = True, delta: float = None
    ) -> Dict[str, Any]:
        """
        记忆被实际使用后的反馈

        当记忆在任务输出中被实际使用时，给予更高权重的反馈。

        Args:
            chunk_id: 被使用的记忆ID
            success: 任务是否成功
            delta: 重要性增量，默认0.15（比召回更高）

        Returns:
            更新后的记忆状态
        """
        if not success:
            logger.debug(
                f"Task failed for chunk {chunk_id}, skipping importance update"
            )
            return self._get_chunk_state(chunk_id)

        delta = delta or self.DEFAULT_USAGE_DELTA
        return self._increment_importance(chunk_id, delta)

    def on_usage_batch(
        self, chunk_ids: List[str], success: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量记忆使用反馈
        """
        results = []
        for chunk_id in chunk_ids:
            try:
                result = self.on_usage(chunk_id, success)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to update usage feedback for {chunk_id}: {e}")
                results.append({"chunk_id": chunk_id, "error": str(e)})
        return results

    def on_task_success(
        self, chunk_ids: List[str], reward: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        整体任务成功的反馈

        当一个任务成功完成时，对所有参与的记忆给予强化。

        Args:
            chunk_ids: 参与任务的记忆ID列表
            reward: 奖励幅度，默认0.2

        Returns:
            更新后的记忆状态列表
        """
        logger.info(
            f"Task success: rewarding {len(chunk_ids)} memories with delta {reward}"
        )
        return self.on_usage_batch(chunk_ids, success=True, delta=reward)

    def on_task_failure(
        self, chunk_ids: List[str], penalty: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        整体任务失败的反馈

        当一个任务失败时，降低相关记忆的重要性。

        Args:
            chunk_ids: 参与任务的记忆ID列表
            penalty: 惩罚幅度，默认0.05

        Returns:
            更新后的记忆状态列表
        """
        logger.info(
            f"Task failure: penalizing {len(chunk_ids)} memories with delta {-penalty}"
        )
        results = []
        for chunk_id in chunk_ids:
            try:
                result = self._increment_importance(chunk_id, -penalty)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to apply penalty to {chunk_id}: {e}")
                results.append({"chunk_id": chunk_id, "error": str(e)})
        return results

    def set_importance(self, chunk_id: str, score: float) -> Dict[str, Any]:
        """
        直接设置重要性评分（用于管理员干预或外部评估）

        Args:
            chunk_id: 记忆ID
            score: 重要性评分，范围0-1

        Returns:
            更新后的记忆状态
        """
        score = max(self.MIN_IMPORTANCE, min(self.MAX_IMPORTANCE, score))
        self.mongo_repo.update_importance_score(chunk_id, score)
        logger.info(f"Set importance for {chunk_id} to {score}")
        return self._get_chunk_state(chunk_id)

    def get_importance(self, chunk_id: str) -> float:
        """
        获取记忆的重要性评分
        """
        chunk = self._get_chunk_state(chunk_id)
        return chunk.get("importance_score", 0.0) if chunk else 0.0

    def get_top_memories(
        self, limit: int = 10, memory_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取重要性最高的记忆列表
        """
        return self.mongo_repo.get_chunks_by_importance(
            memory_type=memory_type, limit=limit
        )

    def get_stale_memories(
        self, days: int = 30, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取长时间未访问的记忆（用于分析衰减效果）
        """
        threshold = datetime.utcnow() - datetime.timedelta(days=days)
        chunks = self.mongo_repo.get_chunks_by_memory_type("episodic")
        stale = [
            c
            for c in chunks
            if c.get("last_reviewed") is None or c["last_reviewed"] < threshold
        ]
        return stale[:limit]

    def _increment_importance(self, chunk_id: str, delta: float) -> Dict[str, Any]:
        """
        内部方法：增加重要性评分
        """
        self.mongo_repo.increment_importance(chunk_id, delta)
        return self._get_chunk_state(chunk_id)

    def _get_chunk_state(self, chunk_id: str) -> Dict[str, Any]:
        """
        获取记忆当前状态
        """
        return self.mongo_repo.get_chunk(chunk_id) or {}

    def close(self):
        """关闭数据库连接"""
        self.mongo_repo.close()


class FeedbackConfig:
    """反馈机制配置"""

    RECALL_DELTA = 0.05
    USAGE_DELTA = 0.15
    TASK_SUCCESS_DELTA = 0.2
    TASK_FAILURE_DELTA = -0.05
    MAX_IMPORTANCE = 1.0
    MIN_IMPORTANCE = 0.0

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "recall_delta": cls.RECALL_DELTA,
            "usage_delta": cls.USAGE_DELTA,
            "task_success_delta": cls.TASK_SUCCESS_DELTA,
            "task_failure_delta": cls.TASK_FAILURE_DELTA,
            "max_importance": cls.MAX_IMPORTANCE,
            "min_importance": cls.MIN_IMPORTANCE,
        }
