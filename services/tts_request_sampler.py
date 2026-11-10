import logging
import random
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from constants import TTS_QUALITY_FAILURE_SAMPLE_RATE, TTS_SAMPLED_RESULT_LIMIT, TTS_VOLATILE_FIELDS

logger = logging.getLogger(__name__)


def _validate(sample_rate: float, failed_sample_rate: float, result_limit: int) -> None:
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be within (0, 1], got {sample_rate}")
    if not 0.0 <= failed_sample_rate <= 1.0:
        raise ValueError(f"failed_sample_rate must be within [0, 1], got {failed_sample_rate}")
    if result_limit < 1:
        raise ValueError(f"result_limit must be at least 1, got {result_limit}")


class TTSRequestSampler:
    def __init__(
        self,
        sample_rate: float = 0.05,
        failed_sample_rate: float = TTS_QUALITY_FAILURE_SAMPLE_RATE,
        clock: Optional[Callable[[], datetime]] = None,
        result_limit: int = TTS_SAMPLED_RESULT_LIMIT,
    ):
        _validate(sample_rate, failed_sample_rate, result_limit)
        self._sample_rate = sample_rate
        self._failed_sample_rate = failed_sample_rate
        self._clock = clock or self._utc_now
        self._strata: Dict[str, int] = {}
        self._day = ""
        self._lock = threading.Lock()
        self._sampled = 0
        self._results: Deque[Dict[str, Any]] = deque(maxlen=result_limit)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def select(self, result: Optional[Dict[str, Any]] = None) -> bool:
        if not self._decide(result):
            return False
        with self._lock:
            self._sampled += 1
        return True

    def _decide(self, result: Optional[Dict[str, Any]]) -> bool:
        if result is None or not result.get("stratum"):
            return random.random() < self._sample_rate
        if self._is_failed(result):
            return random.random() < self._failed_sample_rate
        return self._stratified(str(result["stratum"]))

    @staticmethod
    def _is_failed(result: Dict[str, Any]) -> bool:
        abnormal_status = result.get("status") not in (None, "pass", "skipped")
        return bool(abnormal_status or result.get("duration_anomaly") or result.get("duplicate"))

    def _stratified(self, stratum: str) -> bool:
        day = self._clock().date().isoformat()
        with self._lock:
            if day != self._day:
                self._day = day
                self._strata.clear()
            count = self._strata.get(stratum, 0) + 1
            self._strata[stratum] = count
        return int(count * self._sample_rate) > int((count - 1) * self._sample_rate)

    def record(self, result: Dict[str, Any]) -> None:
        stored = {key: value for key, value in result.items() if key not in TTS_VOLATILE_FIELDS}
        with self._lock:
            self._results.append(stored)

    def results(self, limit: int = TTS_SAMPLED_RESULT_LIMIT) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._results)[-limit:]

    @property
    def sampled_count(self) -> int:
        with self._lock:
            return self._sampled
