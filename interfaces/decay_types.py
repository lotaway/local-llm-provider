from abc import ABC, abstractmethod
from typing import Dict, Any


class IDecayCalculator(ABC):
    @abstractmethod
    def calculate(self, memory: Dict[str, Any]) -> float: ...
