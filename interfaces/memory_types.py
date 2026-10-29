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


class ISensoryCache(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def remember(self, entry_id: str, payload: Dict[str, Any], ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def recall(self, entry_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def forget(self, entry_id: str) -> None: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...


class IWorkingMemory(ABC):
    @abstractmethod
    def focus(self, entry_id: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]: ...

    @abstractmethod
    def recall(self, entry_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def active(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...

