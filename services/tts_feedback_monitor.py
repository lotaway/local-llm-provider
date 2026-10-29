import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSFeedbackMonitor:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def record(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(feedback)
        record.setdefault("regenerate_count", 0)
        self._records.append(record)
        return record

    def distribution(self) -> Dict[str, Any]:
        ratings = Counter(str(item.get("rating")) for item in self._records)
        complaints = Counter(str(item.get("complaint")) for item in self._records if item.get("complaint"))
        return {
            "total": len(self._records),
            "ratings": dict(sorted(ratings.items(), key=lambda pair: pair[0])),
            "complaints": dict(sorted(complaints.items())),
            "avg_regenerate_count": (
                sum(float(item.get("regenerate_count") or 0.0) for item in self._records) / len(self._records)
                if self._records
                else 0.0
            ),
        }

    def complaint_rate(self) -> float:
        if not self._records:
            return 0.0
        return len([item for item in self._records if item.get("complaint")]) / len(self._records)
