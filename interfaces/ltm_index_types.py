from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ILtmIndexRepository(ABC):
    @abstractmethod
    def save_ltm_vector(self, ltm: Dict[str, Any]) -> str: ...

    @abstractmethod
    def index_ltm_keywords(self, ltm: Dict[str, Any]) -> str: ...

    @abstractmethod
    def nearest_similarity(self, ltm: Dict[str, Any]) -> Optional[float]: ...
