import threading
from collections import deque
from typing import Any, Deque, Dict, List

from interfaces.molt_types import IStrategyHistoryStore


class InMemoryStrategyHistoryStore(IStrategyHistoryStore):
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._limit = limit
        self._records: Deque[Dict[str, Any]] = deque(maxlen=limit)
        self._total = 0
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._records.append(dict(record))
            self._total += 1

    def recent(self, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._records)
        return records[-max(1, limit):]

    def count(self) -> int:
        with self._lock:
            return self._total
