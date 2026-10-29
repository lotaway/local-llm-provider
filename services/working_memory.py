from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from interfaces.memory_types import IWorkingMemory


class RingWorkingMemory(IWorkingMemory):
    MIN_CAPACITY = 5
    MAX_CAPACITY = 9

    def __init__(self, capacity: int = 7):
        self._validate_capacity(capacity)
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._evictions = 0

    def focus(self, entry_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        existing = self._find(entry_id)
        if existing is not None:
            return self._rehearse(existing)
        return self._append(entry_id, payload or {})

    def recall(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entry = self._find(entry_id)
        if entry is None:
            return None
        return self._rehearse(entry)

    def active(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._entries]

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self._entries.maxlen,
            "size": len(self._entries),
            "evictions": self._evictions,
            "rehearsals": sum(entry["rehearsal_count"] for entry in self._entries),
        }

    def _validate_capacity(self, capacity: int) -> None:
        if not self.MIN_CAPACITY <= capacity <= self.MAX_CAPACITY:
            raise ValueError(
                f"working memory capacity must be between {self.MIN_CAPACITY} and {self.MAX_CAPACITY}"
            )

    def _find(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return next((entry for entry in self._entries if entry["entry_id"] == entry_id), None)

    def _rehearse(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        entry["rehearsal_count"] += 1
        entry["last_access_at"] = self._now()
        return dict(entry)

    def _append(self, entry_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if len(self._entries) == self._entries.maxlen:
            self._evictions += 1
        entry = {
            "entry_id": entry_id,
            "payload": payload,
            "rehearsal_count": 0,
            "created_at": self._now(),
            "last_access_at": self._now(),
        }
        self._entries.append(entry)
        return dict(entry)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
