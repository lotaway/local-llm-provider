from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IVersionManager(ABC):
    @abstractmethod
    def detect_conflict(self, topic: str, conclusion: str) -> Dict[str, Any]: ...

    @abstractmethod
    def upsert_ltm(self, ltm: Dict[str, Any]) -> str: ...

    @abstractmethod
    def get_version_chain(self, topic: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def rollback_to(self, topic: str, version: int) -> bool: ...
