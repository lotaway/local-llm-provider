import importlib.util
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "repositories" / "document_record.py"
REPOSITORY = ROOT / "repositories" / "mongodb_repository.py"
NEO4J = ROOT / "repositories" / "neo4j_repository.py"


def _load():
    spec = importlib.util.spec_from_file_location("document_record_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fields():
    return {"title": "Guide", "content": "body", "source": "docs/guide.md"}


def test_explicit_document_fields_reach_the_record():
    record = _load().build_document_record("doc-1", _fields(), now=datetime(2026, 1, 1))
    assert record["title"] == "Guide"
    assert record["content"] == "body"
    assert record["source"] == "docs/guide.md"
    assert record["_id"] == "doc-1"


def test_existing_status_and_failure_are_preserved():
    existing = {
        "created_at": "2025-01-01T00:00:00Z",
        "status": {"chunked": True, "embedded": False, "graphed": False},
        "last_error": {"step": "embedded"},
    }
    record = _load().build_document_record("doc-1", _fields(), existing, now=datetime(2026, 1, 1))
    assert record["created_at"] == "2025-01-01T00:00:00Z"
    assert record["status"]["chunked"] is True
    assert record["last_error"]["step"] == "embedded"


def test_default_status_is_not_shared_between_records():
    module = _load()
    first = module.build_document_record("a", _fields(), now=datetime(2026, 1, 1))
    first["status"]["chunked"] = True
    second = module.build_document_record("b", _fields(), now=datetime(2026, 1, 1))
    assert second["status"]["chunked"] is False


def test_save_document_forwards_its_explicit_arguments():
    source = REPOSITORY.read_text(encoding="utf-8")
    assert "record = dict(fields, title=title, content=content, source=source)" in source
    assert "self._build_document_record(doc_id, record)" in source
    assert "self._build_document_record(doc_id, fields)" not in source


def test_subgraph_transaction_has_one_staticmethod_decorator():
    source = NEO4J.read_text(encoding="utf-8")
    assert "    @staticmethod\n    @staticmethod\n" not in source
