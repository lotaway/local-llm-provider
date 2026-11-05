from abc import ABC, abstractmethod


class RuntimeClock(ABC):
    @abstractmethod
    def elapsed_seconds(self) -> float:
        ...
