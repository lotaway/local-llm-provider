from typing import Any, Dict, Iterable, List, Optional

STATUS_KEYS = ("chunked", "embedded", "graphed")


def normalize_status(status: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    current = status or {}
    return {key: bool(current.get(key, False)) for key in STATUS_KEYS}


def is_complete(status: Optional[Dict[str, Any]]) -> bool:
    normalized = normalize_status(status)
    return all(normalized[key] for key in STATUS_KEYS)


def pending_steps(status: Optional[Dict[str, Any]]) -> List[str]:
    normalized = normalize_status(status)
    return [key for key in STATUS_KEYS if not normalized[key]]


def normalize_checksum(checksum: str) -> str:
    return checksum if checksum.startswith("sha256:") else f"sha256:{checksum}"


def checksum_matches(document: Optional[Dict[str, Any]], checksum: str) -> bool:
    if not document:
        return False
    return document.get("checksum", "") == normalize_checksum(checksum)


def needs_rebuild(document: Optional[Dict[str, Any]], checksum: str) -> bool:
    return document is not None and not checksum_matches(document, checksum)


def completed_steps(status: Optional[Dict[str, Any]]) -> Iterable[str]:
    normalized = normalize_status(status)
    return (key for key in STATUS_KEYS if normalized[key])

def failed_step(document: Optional[Dict[str, Any]]) -> Optional[str]:
    error = (document or {}).get("last_error") or {}
    return error.get("step")


def has_failure(document: Optional[Dict[str, Any]]) -> bool:
    return failed_step(document) is not None


def needs_retry(document: Optional[Dict[str, Any]]) -> bool:
    return has_failure(document) or not is_complete((document or {}).get("status"))
