from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IEvolutionDispatcher(ABC):
    @abstractmethod
    def enqueue(self, signal: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def aggregate_and_apply(
        self, context: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ...
