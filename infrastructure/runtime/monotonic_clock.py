import time
from typing import Callable

from domain.runtime.ports import RuntimeClock


class MonotonicRuntimeClock(RuntimeClock):
    def __init__(self, time_source: Callable[[], float] = time.monotonic):
        if not callable(time_source):
            raise TypeError("time_source must be callable")
        self._time_source = time_source
        self._started_at = time_source()

    def elapsed_seconds(self) -> float:
        return self._time_source() - self._started_at
