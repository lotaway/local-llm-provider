from typing import List, Dict
from interfaces.feedback_types import IFeedbackService
from interfaces.memory_types import IMemoryRepository


class FeedbackService(IFeedbackService):
    RECALL_DELTA = 0.05
    USAGE_DELTA = 0.15
    MAX_SCORE = 1.0
    MIN_SCORE = 0.0

    def __init__(self, memory_repo: IMemoryRepository):
        self._repo = memory_repo

    def on_recall(self, chunk_id: str) -> Dict:
        return self._increment(chunk_id, self.RECALL_DELTA)

    def on_recall_batch(self, chunk_ids: List[str]) -> List[Dict]:
        return [self.on_recall(cid) for cid in chunk_ids if cid]

    def on_usage(self, chunk_id: str, success: bool) -> Dict:
        if not success:
            return self._repo.get_by_id(chunk_id) or {}
        return self._increment(chunk_id, self.USAGE_DELTA)

    def on_task_result(self, chunk_ids: List[str], success: bool) -> List[Dict]:
        delta = self.USAGE_DELTA if success else -0.05
        return [self._increment(cid, delta) for cid in chunk_ids if cid]

    def _increment(self, chunk_id: str, delta: float) -> Dict:
        new_score = self._repo.get_by_id(chunk_id).get("importance_score", 0) + delta
        new_score = max(self.MIN_SCORE, min(self.MAX_SCORE, new_score))
        self._repo.update_importance(chunk_id, new_score)
        self._repo.mark_reviewed(chunk_id)
        return self._repo.get_by_id(chunk_id)
