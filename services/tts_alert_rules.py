from typing import Any, Dict, Optional


def should_alert(
    result: Dict[str, Any],
    wer_threshold: float,
    dnsmos_threshold: float,
    similarity_threshold: float,
) -> bool:
    if result.get("status") == "reject":
        return True
    if _exceeds(result.get("wer"), wer_threshold):
        return True
    if _below((result.get("dnsmos") or {}).get("overall"), dnsmos_threshold):
        return True
    return _below(result.get("speaker_similarity"), similarity_threshold)


def _exceeds(value: Optional[Any], threshold: float) -> bool:
    return value is not None and float(value) > threshold


def _below(value: Optional[Any], threshold: float) -> bool:
    return value is not None and float(value) < threshold
