from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List


class MoltJob(str, Enum):
    DECAY = "decay"
    ABSTRACTION = "abstraction"
    EVALUATION = "evaluation"


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
