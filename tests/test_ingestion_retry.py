from services.ingestion_checkpoint import failed_step, has_failure, needs_retry


def _document(status=None, error=None):
    document = {"status": status or {"chunked": True, "embedded": True, "graphed": True}}
    if error is not None:
        document["last_error"] = error
    return document


def test_failed_step_reads_the_persisted_error():
    document = _document(error={"step": "embedded", "message": "boom"})
    assert failed_step(document) == "embedded"
    assert has_failure(document) is True


def test_needs_retry_for_incomplete_or_failed_documents():
    incomplete = _document(status={"chunked": True, "embedded": False, "graphed": False})
    assert needs_retry(incomplete) is True
    assert needs_retry(_document(error={"step": "graphed"})) is True


def test_complete_document_without_error_does_not_need_retry():
    assert needs_retry(_document()) is False
    assert has_failure(_document()) is False
