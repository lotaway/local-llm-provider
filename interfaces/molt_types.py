from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List


class MoltJob(str, Enum):
    DECAY = "decay"
    ABSTRACTION = "abstraction"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class MoltStrategyConfig:
    recall_hit_rate_threshold: float = 0.3
    min_abstraction_rate: float = 0.01
    recall_step: float = 0.05
    decay_step: float = 0.005
    abstraction_step: float = 0.05
    recall_floor: float = 0.3
    decay_ceiling: float = 0.05
    abstraction_floor: float = 0.4
    low_importance_threshold: float = 0.4
    initial_recall_threshold: float = 0.7
    initial_decay_rate: float = 0.01
    initial_abstraction_threshold: float = 0.6
    history_limit: int = 50

    def __post_init__(self) -> None:
        rates = {
            "recall_hit_rate_threshold": self.recall_hit_rate_threshold,
            "min_abstraction_rate": self.min_abstraction_rate,
            "recall_floor": self.recall_floor,
            "decay_ceiling": self.decay_ceiling,
            "abstraction_floor": self.abstraction_floor,
            "low_importance_threshold": self.low_importance_threshold,
            "initial_recall_threshold": self.initial_recall_threshold,
            "initial_decay_rate": self.initial_decay_rate,
            "initial_abstraction_threshold": self.initial_abstraction_threshold,
        }
        for name, value in rates.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        steps = {
            "recall_step": self.recall_step,
            "decay_step": self.decay_step,
            "abstraction_step": self.abstraction_step,
        }
        for name, value in steps.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.history_limit < 1:
            raise ValueError(f"history_limit must be at least 1, got {self.history_limit}")


class IStrategyHistoryStore(ABC):
    @abstractmethod
    def append(self, record: Dict[str, Any]) -> None: ...

    @abstractmethod
    def recent(self, limit: int) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...


@dataclass(frozen=True)
class MoltScheduleConfig:
    decay_interval_seconds: int = 3600
    abstraction_interval_seconds: int = 21600
    evaluation_interval_seconds: int = 86400
    max_retries: int = 2
    retry_backoff_seconds: float = 5.0
    max_concurrency: int = 1

    def interval_for(self, job: MoltJob) -> int:
        return {
            MoltJob.DECAY: self.decay_interval_seconds,
            MoltJob.ABSTRACTION: self.abstraction_interval_seconds,
            MoltJob.EVALUATION: self.evaluation_interval_seconds,
        }[job]


class IMoltController(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

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
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...
