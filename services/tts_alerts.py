import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from constants import (
    TTS_ALERTS_LIMIT,
    TTS_DNSMOS_THRESHOLD,
    TTS_SIMILARITY_THRESHOLD,
    TTS_VOLATILE_FIELDS,
    TTS_WER_THRESHOLD,
)
from interfaces.tts_types import ITTSNotifier
from services.tts_alert_rules import should_alert as breaches_quality_rules

logger = logging.getLogger(__name__)


class TTSAlertService:
    def __init__(
        self,
        notifier: Optional[ITTSNotifier] = None,
        wer_threshold: float = TTS_WER_THRESHOLD,
        dnsmos_threshold: float = TTS_DNSMOS_THRESHOLD,
        similarity_threshold: float = TTS_SIMILARITY_THRESHOLD,
        limit: int = TTS_ALERTS_LIMIT,
    ):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._notifier = notifier
        self._wer_threshold = wer_threshold
        self._dnsmos_threshold = dnsmos_threshold
        self._similarity_threshold = similarity_threshold
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def should_alert(self, result: Dict[str, Any]) -> bool:
        return breaches_quality_rules(
            result, self._wer_threshold, self._dnsmos_threshold, self._similarity_threshold
        )

    def check(self, results: List[Dict[str, Any]]) -> None:
        for result in results:
            if not self.should_alert(result):
                continue
            alert = _sanitize(result)
            with self._lock:
                self._alerts.append(alert)
            self._notify(alert)

    def _notify(self, alert: Dict[str, Any]) -> None:
        if self._notifier is not None:
            self._notifier.notify(alert)

    def alerts(self, limit: int = TTS_ALERTS_LIMIT) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def reset(self) -> None:
        with self._lock:
            self._alerts.clear()


def _sanitize(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key not in TTS_VOLATILE_FIELDS}
