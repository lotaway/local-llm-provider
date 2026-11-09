import threading
from datetime import datetime, timezone
from typing import Callable, Optional


class DailyEvaluationQuota:
    def __init__(self, limit: int, clock: Optional[Callable[[], datetime]] = None) -> None:
        if limit < 0:
            raise ValueError(f"limit must not be negative, got {limit}")
        self._limit = limit
        self._clock = clock or self._utc_now
        self._lock = threading.Lock()
        self._day = ""
        self._used = 0

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @property
    def limit(self) -> int:
        return self._limit

    def reserve(self) -> bool:
        if self._limit == 0:
            return True
        day = self._clock().date().isoformat()
        with self._lock:
            if self._day != day:
                self._day = day
                self._used = 0
            if self._used >= self._limit:
                return False
            self._used += 1
            return True
