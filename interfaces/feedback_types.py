from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IFeedbackService(ABC):
    @abstractmethod
    def on_recall(self, chunk_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def on_recall_batch(self, chunk_ids: List[str]) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def on_usage(self, chunk_id: str, success: bool) -> Dict[str, Any]: ...

    @abstractmethod
    def on_task_result(
        self, chunk_ids: List[str], success: bool
    ) -> List[Dict[str, Any]]: ...
