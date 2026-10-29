import logging
import statistics
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSTrendStatistics:
    def __init__(self, window: int = 1000):
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self._results: deque = deque(maxlen=window)

    def add(self, result: Dict[str, Any]) -> None:
        self._results.append(dict(result))

    def wer_statistics(self) -> Dict[str, float]:
        values = [float(item.get("wer", 0.0) or 0.0) for item in self._results]
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "p90": self._percentile(values, 0.90),
            "p95": self._percentile(values, 0.95),
            "count": float(len(values)),
        }

    def dnsmos_statistics(self) -> Dict[str, float]:
        values = [
            float((item.get("dnsmos") or {}).get("overall", 0.0) or 0.0)
            for item in self._results
        ]
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "p90": self._percentile(values, 0.90),
            "p95": self._percentile(values, 0.95),
            "count": float(len(values)),
        }

    def similarity_statistics(self) -> Dict[str, float]:
        values = [
            float((item.get("speaker_similarity") or item.get("similarity") or 0.0))
            for item in self._results
        ]
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "p90": self._percentile(values, 0.90),
            "p95": self._percentile(values, 0.95),
            "count": float(len(values)),
        }

    def failure_rate(self) -> float:
        if not self._results:
            return 0.0
        failures = sum(1 for item in self._results if item.get("status") == "reject")
        return failures / len(self._results)

    def rolling_mean(self, key: str, window: int = 100) -> float:
        items = list(self._results)[-window:]
        values = [float(item.get(key, 0.0) or 0.0) for item in items]
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _percentile(values: List[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1)))))
        return float(ordered[index])
