from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class IMemoryQuery(ABC):
    @abstractmethod
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def get_by_type(self, memory_type: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_by_importance(
        self, memory_type: str, limit: int
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_low_importance(
        self, threshold: float, limit: int
    ) -> List[Dict[str, Any]]: ...


class IMemoryLifecycle(ABC):
    @abstractmethod
    def update_importance(self, chunk_id: str, score: float) -> None: ...

    @abstractmethod
    def increment_importance(self, chunk_id: str, delta: float) -> None: ...

    @abstractmethod
    def mark_reviewed(self, chunk_id: str) -> None: ...

    @abstractmethod
    def mark_discarded(self, chunk_id: str) -> None: ...


class IMemoryRepository(IMemoryQuery, IMemoryLifecycle, ABC):
    pass
