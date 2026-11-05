import logging
from typing import Any, Dict, List, Optional

from constants import (
    TTS_ALERTS_LIMIT,
    TTS_DNSMOS_THRESHOLD,
    TTS_SIMILARITY_THRESHOLD,
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
    ):
        self._notifier = notifier
        self._wer_threshold = wer_threshold
        self._dnsmos_threshold = dnsmos_threshold
        self._similarity_threshold = similarity_threshold
        self._alerts: List[Dict[str, Any]] = []

    def should_alert(self, result: Dict[str, Any]) -> bool:
        return breaches_quality_rules(
            result, self._wer_threshold, self._dnsmos_threshold, self._similarity_threshold
        )

    def check(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        alerts = [result for result in results if self.should_alert(result)]
        for result in alerts:
            self._alerts.append(dict(result))
            if self._notifier is not None:
                self._notifier.notify(result)
        return alerts

    def alerts(self, limit: int = TTS_ALERTS_LIMIT) -> List[Dict[str, Any]]:
        return list(self._alerts[:limit])

    def reset(self) -> None:
        self._alerts = []
