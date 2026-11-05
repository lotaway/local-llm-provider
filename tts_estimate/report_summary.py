from typing import Any, Dict, List


def build_report(result: Any) -> Dict[str, Any]:
    return {
        "wer": result.metrics.wer,
        "dnsmos": {
            "overall": result.metrics.dnsmos_overall,
            "signal": result.metrics.dnsmos_signal,
            "background": result.metrics.dnsmos_background,
        },
        "speaker_similarity": result.metrics.speaker_similarity,
        "status": "pass" if result.is_passed else "reject",
    }


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = [report.get("status") for report in reports]
    return {
        "total": len(reports),
        "passed": statuses.count("pass"),
        "rejected": statuses.count("reject"),
        "max_wer": max((float(r.get("wer") or 0.0) for r in reports), default=0.0),
    }
