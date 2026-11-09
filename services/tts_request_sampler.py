import logging
import random
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from constants import TTS_QUALITY_FAILURE_SAMPLE_RATE

logger = logging.getLogger(__name__)


class TTSRequestSampler:
    def __init__(
        self,
        sample_rate: float = 0.05,
        failed_sample_rate: float = TTS_QUALITY_FAILURE_SAMPLE_RATE,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be within (0, 1], got {sample_rate}")
        if not 0.0 <= failed_sample_rate <= 1.0:
            raise ValueError(f"failed_sample_rate must be within [0, 1], got {failed_sample_rate}")
        self._sample_rate = sample_rate
        self._failed_sample_rate = failed_sample_rate
        self._clock = clock or self._utc_now
        self._strata: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._sampled = 0
        self._results: List[Dict[str, Any]] = []

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def should_sample(self, result: Optional[Dict[str, Any]] = None) -> bool:
        if result is None or not result.get("stratum"):
            sample = random.random() < self._sample_rate
        elif self._is_failed(result):
            sample = random.random() < self._failed_sample_rate
        else:
            sample = self._stratified(str(result["stratum"]))
        if sample:
            self._sampled += 1
        return sample

    @staticmethod
    def _is_failed(result: Dict[str, Any]) -> bool:
        abnormal_status = result.get("status") not in (None, "pass", "skipped")
        return bool(abnormal_status or result.get("duration_anomaly") or result.get("duplicate"))

    def _stratified(self, stratum: str) -> bool:
        key = f"{self._clock().date().isoformat()}:{stratum}"
        with self._lock:
            count = self._strata.get(key, 0) + 1
            self._strata[key] = count
        return int(count * self._sample_rate) > int((count - 1) * self._sample_rate)

    def record(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self._results.append(dict(result))
        return dict(result)

    def results(self, limit: int = 1000) -> List[Dict[str, Any]]:
        return list(self._results[:limit])

    @property
    def sampled_count(self) -> int:
        return self._sampled
