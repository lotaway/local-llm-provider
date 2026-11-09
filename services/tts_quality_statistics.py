import logging
import statistics
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSQualityStatistics:
    def __init__(
        self,
        window: int = 1000,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self._results: deque = deque(maxlen=window)
        self._days: deque = deque(maxlen=window)
        self._clock = clock or self._utc_now

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def add(self, result: Dict[str, Any]) -> None:
        self._results.append(dict(result))
        self._days.append(self._clock().date().isoformat())

    def wer_statistics(self) -> Dict[str, float]:
        return self._summary(list(self._results), lambda item: item.get("wer"))

    def dnsmos_statistics(self) -> Dict[str, float]:
        return self._summary(list(self._results), lambda item: (item.get("dnsmos") or {}).get("overall"))

    def similarity_statistics(self) -> Dict[str, float]:
        return self._summary(list(self._results), lambda item: item.get("speaker_similarity", item.get("similarity")))

    def failure_rate(self) -> float:
        return self._ratio(list(self._results), lambda item: item.get("status") == "reject")

    def rolling_mean(self, key: str, window: int = 100) -> float:
        items = list(self._results)[-window:]
        values = [float(item.get(key, 0.0) or 0.0) for item in items]
        return statistics.mean(values) if values else 0.0

    def daily_statistics(self, day: Optional[str] = None) -> Dict[str, Any]:
        target = day or self._clock().date().isoformat()
        records = [item for item, recorded in zip(self._results, self._days) if recorded == target]
        return {
            "day": target,
            "count": float(len(records)),
            "wer": self._summary(records, lambda item: item.get("wer")),
            "dnsmos": self._summary(records, lambda item: (item.get("dnsmos") or {}).get("overall")),
            "speaker_similarity": self._summary(records, lambda item: item.get("speaker_similarity", item.get("similarity"))),
            "failure_rate": self._ratio(records, lambda item: item.get("status") == "reject"),
            "anomaly_rate": self._ratio(records, self._is_anomaly),
        }

    def daily_breakdown(self, limit: int = 7) -> List[Dict[str, Any]]:
        if limit < 1:
            return []
        days = sorted({value for value in self._days})[-limit:]
        return [self.daily_statistics(day) for day in days]

    @staticmethod
    def _is_anomaly(item: Dict[str, Any]) -> bool:
        abnormal_status = item.get("status") not in ("pass", "skipped", None)
        return bool(abnormal_status or item.get("duration_anomaly") or item.get("duplicate"))

    @staticmethod
    def _summary(
        records: List[Dict[str, Any]], extractor: Callable[[Dict[str, Any]], Any]
    ) -> Dict[str, float]:
        values = [float(value) for item in records if (value := extractor(item)) is not None]
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "p90": TTSQualityStatistics._percentile(values, 0.90),
            "p95": TTSQualityStatistics._percentile(values, 0.95),
            "count": float(len(values)),
        }

    @staticmethod
    def _ratio(
        records: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]
    ) -> float:
        if not records:
            return 0.0
        return sum(1 for item in records if predicate(item)) / len(records)

    @staticmethod
    def _percentile(values: List[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1)))))
        return float(ordered[index])
