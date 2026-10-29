from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IMoltController(ABC):
    @abstractmethod
    def run_decay_cycle(self) -> Dict[str, Any]: ...

    @abstractmethod
    def run_abstraction_cycle(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def run_full_cycle(self) -> Dict[str, Any]: ...

    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]: ...

    @abstractmethod
    def evaluate_and_adjust(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_strategy(self) -> Dict[str, float]: ...


class IMoltScheduler(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...
