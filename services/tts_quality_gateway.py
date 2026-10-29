import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TTSSpeechQualityGateway:
    def __init__(self, evaluator: Any = None):
        self._evaluator = evaluator

    def evaluate(self, ref_text: str, ref_audio: bytes, gen_audio: bytes) -> Dict[str, Any]:
        if self._evaluator is None:
            return {"status": "rejected", "reason": "evaluator_unavailable"}
        result = self._evaluator.evaluate(ref_text, ref_audio, gen_audio)
        report = self._to_report(result)
        logger.info(
            "tts quality evaluated",
            extra={"status": report.get("status"), "wer": report.get("wer")},
        )
        return report

    def should_block(self, report: Dict[str, Any]) -> bool:
        return report.get("status") != "pass"

    def _to_report(self, result: Any) -> Dict[str, Any]:
        metrics = getattr(result, "metrics", None)
        status = "pass" if bool(getattr(result, "is_passed", True)) else "reject"
        return {
            "status": status,
            "wer": float(getattr(metrics, "wer", 0.0) or 0.0),
            "dnsmos": {
                "overall": float(getattr(metrics, "dnsmos_overall", 0.0) or 0.0),
                "signal": float(getattr(metrics, "dnsmos_signal", 0.0) or 0.0),
                "background": float(getattr(metrics, "dnsmos_background", 0.0) or 0.0),
            },
            "speaker_similarity": float(getattr(result, "speaker_similarity", 0.0) or 0.0),
        }
