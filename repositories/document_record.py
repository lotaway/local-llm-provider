from datetime import datetime
from typing import Any, Dict, Optional

OPTIONAL_FIELDS = ("source", "author", "summary", "path", "filename")
DEFAULT_STATUS = {"chunked": False, "embedded": False, "graphed": False}


def build_document_record(
    doc_id: str,
    fields: Dict[str, Any],
    existing: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    stamp = (now or datetime.utcnow()).isoformat() + "Z"
    stored = existing or {}
    return _with_optional_fields(_base_record(doc_id, fields, stored, stamp), fields, stored)


def _base_record(doc_id: str, fields: Dict[str, Any], stored: Dict[str, Any], stamp: str) -> Dict[str, Any]:
    return {
        "_id": doc_id,
        "title": fields["title"],
        "content": fields["content"],
        "format": fields.get("format", "markdown"),
        "created_at": stored.get("created_at", stamp),
        "updated_at": stamp,
        "checksum": fields.get("checksum") or "",
        "metadata": fields.get("metadata") or {},
        "status": stored.get("status", dict(DEFAULT_STATUS)),
    }


def _with_optional_fields(record: Dict[str, Any], fields: Dict[str, Any], stored: Dict[str, Any]) -> Dict[str, Any]:
    for key in OPTIONAL_FIELDS:
        record[key] = fields.get(key) or ""
    if stored.get("last_error"):
        record["last_error"] = stored["last_error"]
    return record
