import logging
from typing import Any, Dict

from interfaces.tts_types import ITTSNotifier

logger = logging.getLogger(__name__)


class StructuredLogNotifier(ITTSNotifier):
    def notify(self, result: Dict[str, Any]) -> None:
        logger.warning(
            "tts quality alert",
            extra={
                "status": result.get("status"),
                "wer": result.get("wer"),
                "dnsmos": result.get("dnsmos"),
                "speaker_similarity": result.get("speaker_similarity"),
            },
        )
