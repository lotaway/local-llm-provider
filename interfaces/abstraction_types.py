from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IAbstractionEngine(ABC):
    @abstractmethod
    def can_abstraction_trigger(self, topic: str) -> bool: ...

    @abstractmethod
    def abstract_to_ltm(self, topic: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def auto_discover_and_abstract(self) -> List[Dict[str, Any]]: ...
