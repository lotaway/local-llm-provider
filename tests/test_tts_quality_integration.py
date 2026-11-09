import asyncio
import base64
import io
import sys
import threading
import time
import types
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from constants import TTS_MONGO_COLLECTION_QUALITY
from infrastructure.tts.quality_evaluator_adapter import ProductionTTSQualityEvaluator
from infrastructure.tts.quality_record_repository import MongoTTSQualityRecordRepository
from infrastructure.tts.wav_bytes import merge_wav_segments
from services.tts_alerts import TTSAlertService
from services.tts_quality_gateway import TTSQualityGateway
from services.tts_quality_observer import TTSQualityObserver
from services.tts_quality_quota import DailyEvaluationQuota
from services.tts_quality_statistics import TTSQualityStatistics
from services.tts_request_sampler import TTSRequestSampler

ROOT = Path(__file__).resolve().parents[1]


def _wav(frames: bytes, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


class _Result:
    def __init__(self, similarity: Optional[float], is_passed: bool = True) -> None:
        self.metrics = SimpleNamespace(
            wer=0.05,
            dnsmos_overall=4.1,
            dnsmos_signal=4.0,
            dnsmos_background=4.2,
            speaker_similarity=similarity,
        )
        self.is_passed = is_passed


class _Evaluator:
    def __init__(self, similarity: Optional[float] = 0.91) -> None:
        self.similarity = similarity
        self.thresholds: Dict[str, float] = {}

    def evaluate(self, ref_text, ref_audio, gen_audio, thresholds=None) -> Any:
        self.thresholds = thresholds or {}
        return _Result(self.similarity)


class _FailingEvaluator:
    def evaluate(self, *args, **kwargs) -> Any:
        raise RuntimeError("model unavailable")


class _BlockingEvaluator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def evaluate(self, *args, **kwargs) -> Any:
        self.started.set()
        self.release.wait(2.0)
        return _Result(0.91)


class _InlineExecutor:
    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


class _PendingExecutor:
    def submit(self, fn, *args, **kwargs) -> Future:
        return Future()


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, ref_text, ref_audio, gen_audio, thresholds=None) -> Any:
        with wave.open(gen_audio, "rb") as handle:
            frames = handle.getnframes()
        self.calls.append((ref_text, ref_audio, gen_audio, thresholds, frames))
        return _Result(0.9)


class _FakeAdapter:
    def __init__(self, chunk: bytes, first_chunk_ms: Optional[float] = None) -> None:
        self._chunk = chunk
        self._first_chunk_ms = first_chunk_ms

    async def generate_speech_stream(
        self, text, emotion_params, speaker_id, callback=None, user_id=""
    ):
        if callback is not None and self._first_chunk_ms is not None:
            callback({"type": "first_chunk", "elapsed_ms": self._first_chunk_ms})
        yield base64.b64encode(self._chunk).decode("ascii")


class _FakeRecordStore:
    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error
        self.records = []

    def save(self, record: Dict[str, Any]) -> None:
        if self.error is not None:
            raise self.error
        self.records.append(dict(record))


class _FakeCollection:
    def __init__(self) -> None:
        self.inserted = []

    def insert_one(self, payload: Dict[str, Any]) -> None:
        self.inserted.append(payload)


class _FakeDatabase:
    def __init__(self, create_error: Optional[Exception] = None) -> None:
        self.create_error = create_error
        self.created = []
        self.collections: Dict[str, _FakeCollection] = {}

    def create_collection(self, name: str, **options: Any) -> _FakeCollection:
        if self.create_error is not None:
            raise self.create_error
        self.created.append((name, options))
        return self._collection(name)

    def _collection(self, name: str) -> _FakeCollection:
        return self.collections.setdefault(name, _FakeCollection())

    def __getitem__(self, name: str) -> _FakeCollection:
        return self._collection(name)


class _FakeClient:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> _FakeDatabase:
        return self.database


class _MutableClock:
    def __init__(self, day: int) -> None:
        self.day = day

    def __call__(self) -> datetime:
        return datetime(2026, 11, self.day, tzinfo=timezone.utc)


async def _collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


def _observer(sample_rate=1.0, evaluator=None, executor=None, max_pending=8):
    trends = TTSQualityStatistics()
    sampler = TTSRequestSampler(sample_rate=sample_rate)
    alerts = TTSAlertService()
    gateway = TTSQualityGateway(evaluator or _Evaluator())
    executor = executor or _InlineExecutor()
    observer = TTSQualityObserver(gateway, trends, sampler, alerts, executor=executor, max_pending=max_pending)
    return observer, trends, sampler, alerts


def _stub_composition_dependencies(monkeypatch):
    package = types.ModuleType("model_providers")
    package.__path__ = [str(ROOT / "model_providers")]
    monkeypatch.setitem(sys.modules, "model_providers", package)
    adapter_module = types.ModuleType("model_providers.tts.style_tts2_adapter")
    adapter_module.StyleTTS2Adapter = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "model_providers.tts.style_tts2_adapter", adapter_module)
    preferences_module = types.ModuleType("infrastructure.tts.preference_collection")
    preferences_module.build_preference_collection = lambda: object()
    monkeypatch.setitem(sys.modules, "infrastructure.tts.preference_collection", preferences_module)
    speaker_module = types.ModuleType("infrastructure.tts.speaker_repository")
    speaker_module.MongoSpeakerRepository = lambda: object()
    monkeypatch.setitem(sys.modules, "infrastructure.tts.speaker_repository", speaker_module)


def _lightweight_result() -> Dict[str, Any]:
    return {"status": "pass", "text": "hello", "duration_seconds": 1.0}


def test_merge_wav_segments_concatenates_streamed_chunks():
    payload = _wav(b"\x01\x00" * 100) + _wav(b"\x02\x00" * 50)
    with wave.open(io.BytesIO(merge_wav_segments(payload)), "rb") as handle:
        assert handle.getnframes() == 150
        assert handle.readframes(150) == b"\x01\x00" * 100 + b"\x02\x00" * 50


def test_merge_wav_segments_rejects_mixed_formats():
    payload = _wav(b"\x00\x00" * 10) + _wav(b"\x00\x00" * 10, rate=8000)
    with pytest.raises(ValueError):
        merge_wav_segments(payload)


def test_merge_wav_segments_rejects_non_wav_payload():
    with pytest.raises(ValueError):
        merge_wav_segments(b"not audio")


def test_gateway_report_reads_similarity_from_metrics():
    evaluator = _Evaluator(similarity=0.91)
    report = TTSQualityGateway(evaluator).evaluate("hello", b"ref", b"gen")
    assert report["speaker_similarity"] == 0.91
    assert report["dnsmos"]["overall"] == 4.1
    assert evaluator.thresholds["wer_max"] == 0.08


def test_gateway_without_evaluator_allows_the_response():
    gateway = TTSQualityGateway()
    report = gateway.evaluate("hello", b"", b"gen")
    assert report["status"] == "skipped"
    assert gateway.should_block(report) is False


def test_adapter_materializes_bytes_and_loads_models_once():
    recorder = _RecordingEvaluator()
    created = []

    def factory():
        created.append(recorder)
        return recorder

    adapter = ProductionTTSQualityEvaluator(evaluator_factory=factory)
    payload = _wav(b"\x00\x00" * 20)
    adapter.evaluate("hello", payload, payload, {"wer_max": 0.08})
    adapter.evaluate("hello", b"", payload, {"wer_max": 0.08})
    assert len(created) == 1
    assert recorder.calls[0][1].endswith("reference.wav")
    assert recorder.calls[0][2].endswith("generated.wav")
    assert recorder.calls[0][4] == 20
    assert recorder.calls[1][1] == ""


def test_evaluator_without_reference_audio_skips_similarity_check():
    from tts_estimate.application.evaluator_service import TTSQualityEvaluator

    asr, mos, speaker, wer_calc = (MagicMock() for _ in range(4))
    asr.transcribe.return_value = "hello"
    wer_calc.compute_wer.return_value = 0.0
    mos.calculate_score.return_value = {"overall": 4.2, "signal": 4.2, "background": 4.2}
    evaluator = TTSQualityEvaluator(asr, mos, speaker, wer_calc)
    thresholds = {"wer_max": 0.08, "dnsmos_min": 3.8, "similarity_min": 0.78}
    result = evaluator.evaluate("hello", "", "gen.wav", thresholds)
    assert result.metrics.speaker_similarity is None
    assert result.is_passed is True
    speaker.calculate_similarity.assert_not_called()


def test_statistics_ignore_records_without_quality_metrics():
    statistics = TTSQualityStatistics()
    statistics.add({"status": "pass", "text": "hello"})
    statistics.add({"status": "pass", "wer": 0.2, "dnsmos": {"overall": 4.0}, "speaker_similarity": None})
    assert statistics.wer_statistics()["count"] == 1.0
    assert statistics.wer_statistics()["mean"] == 0.2
    assert statistics.similarity_statistics()["count"] == 0.0


def test_sampled_request_records_metrics_without_blocking():
    observer, trends, sampler, _ = _observer()
    observer.observe(_lightweight_result(), b"audio", None)
    assert trends.wer_statistics()["mean"] == 0.05
    assert trends.dnsmos_statistics()["mean"] == 4.1
    assert trends.similarity_statistics()["mean"] == 0.91
    assert sampler.results()[0]["evaluated"] is True
    assert sampler.sampled_count == 1


def test_background_thread_records_metrics_after_the_response():
    blocker = _BlockingEvaluator()
    with ThreadPoolExecutor(max_workers=1) as pool:
        observer, trends, _, _ = _observer(evaluator=blocker, executor=pool)
        observer.observe(_lightweight_result(), b"audio", None)
        assert blocker.started.wait(2.0) is True
        assert trends.wer_statistics()["count"] == 0.0
        blocker.release.set()
        deadline = time.monotonic() + 2.0
        while trends.wer_statistics()["count"] == 0.0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert trends.wer_statistics()["mean"] == 0.05


def test_unsampled_request_keeps_only_lightweight_record(monkeypatch):
    monkeypatch.setattr("services.tts_request_sampler.random.random", lambda: 0.99)
    observer, trends, sampler, _ = _observer(sample_rate=0.05)
    observer.observe(_lightweight_result(), b"audio", None)
    assert trends.wer_statistics()["count"] == 0.0
    assert trends.failure_rate() == 0.0
    assert sampler.sampled_count == 0


def test_evaluation_failure_never_raises_and_is_reported():
    observer, trends, _, alerts = _observer(evaluator=_FailingEvaluator())
    observer.observe(_lightweight_result(), b"audio", None)
    assert alerts.alerts()[0]["reason"] == "evaluation_failed"
    assert trends.failure_rate() == 0.0


def test_queue_full_drops_evaluation_without_blocking():
    observer, trends, _, _ = _observer(executor=_PendingExecutor(), max_pending=1)
    observer.observe(_lightweight_result(), b"audio", None)
    observer.observe(_lightweight_result(), b"audio", None)
    assert trends.wer_statistics()["count"] == 0.0


def test_stream_never_blocks_and_metrics_contain_quality_scores(monkeypatch):
    lightweight_package = types.ModuleType("model_providers")
    lightweight_package.__path__ = [str(ROOT / "model_providers")]
    monkeypatch.setitem(sys.modules, "model_providers", lightweight_package)
    from services.tts_service import TTSService
    observer, trends, sampler, alerts = _observer()
    chunk = _wav(b"\x00\x00" * 20)
    service = TTSService(
        adapter=_FakeAdapter(chunk),
        quality_observer=observer,
        trends=trends,
        sampling_monitor=sampler,
        alerts=alerts,
    )
    assert asyncio.run(_collect(service.stream("hello"))) == chunk
    metrics = service.metrics()
    assert metrics["wer"]["mean"] == 0.05
    assert metrics["dnsmos"]["mean"] == 4.1
    assert metrics["sampled_requests"] == 1
    assert metrics["daily"][-1]["wer"]["mean"] == 0.05


def test_composition_root_wires_the_production_observer():
    source = (ROOT / "bootstrap" / "tts_composition.py").read_text(encoding="utf-8")
    assert "ProductionTTSQualityEvaluator()" in source
    assert "quality_observer=observer" in source
    assert "TTSQualityRejected" not in source


def test_composition_root_builds_a_service_with_the_observer(monkeypatch):
    _stub_composition_dependencies(monkeypatch)
    from bootstrap.tts_composition import build_tts_service

    service = build_tts_service()
    observer = service.quality_observer
    assert observer is not None
    assert observer._trends is service.trends
    assert observer._sampler is service.sampling_monitor
    assert observer._alerts is service.alerts
    assert isinstance(observer._gateway._evaluator, ProductionTTSQualityEvaluator)
    assert isinstance(observer._record_store, MongoTTSQualityRecordRepository)
    assert isinstance(observer._quota, DailyEvaluationQuota)
    observer._executor.shutdown(wait=False)


def test_controller_forwards_optional_reference_audio():
    source = (ROOT / "controllers" / "tts_controller.py").read_text(encoding="utf-8")
    assert "reference_audio: Optional[str] = None" in source
    assert "reference_audio=reference_audio" in source
    assert "TTSQualityRejected" not in source


def _service_class(monkeypatch):
    lightweight_package = types.ModuleType("model_providers")
    lightweight_package.__path__ = [str(ROOT / "model_providers")]
    monkeypatch.setitem(sys.modules, "model_providers", lightweight_package)
    from services.tts_service import TTSService

    return TTSService


def test_first_chunk_latency_is_recorded_as_an_observation_only(monkeypatch):
    service_class = _service_class(monkeypatch)
    observer, trends, sampler, alerts = _observer()
    chunk = _wav(b"\x00\x00" * 20)
    service = service_class(
        adapter=_FakeAdapter(chunk, first_chunk_ms=123.0),
        quality_observer=observer,
        trends=trends,
        sampling_monitor=sampler,
        alerts=alerts,
    )
    assert asyncio.run(_collect(service.stream("hello"))) == chunk
    metrics = service.metrics()
    assert metrics["first_chunk"]["mean"] == 123.0
    assert metrics["first_chunk"]["count"] == 1.0
    assert alerts.alerts() == []


def test_daily_quota_degrades_without_blocking():
    trends = TTSQualityStatistics()
    sampler = TTSRequestSampler(sample_rate=1.0)
    alerts = TTSAlertService()
    observer = TTSQualityObserver(
        TTSQualityGateway(_Evaluator()), trends, sampler, alerts,
        executor=_InlineExecutor(), quota=DailyEvaluationQuota(1),
    )
    observer.observe(_lightweight_result(), b"audio", None)
    observer.observe(_lightweight_result(), b"audio", None)
    assert sampler.sampled_count == 2
    assert trends.wer_statistics()["count"] == 1.0
    assert alerts.alerts() == []


def test_daily_quota_resets_on_a_new_utc_day():
    clock = _MutableClock(7)
    quota = DailyEvaluationQuota(1, clock=clock)
    assert quota.reserve() is True
    assert quota.reserve() is False
    clock.day = 8
    assert quota.reserve() is True


def test_observer_persists_evaluated_record_through_the_store():
    collection = _FakeCollection()
    store = MongoTTSQualityRecordRepository(collection=collection, clock=_MutableClock(9))
    trends = TTSQualityStatistics()
    observer = TTSQualityObserver(
        TTSQualityGateway(_Evaluator()), trends, TTSRequestSampler(sample_rate=1.0),
        TTSAlertService(), executor=_InlineExecutor(), record_store=store,
    )
    observer.observe({**_lightweight_result(), "stratum": "user-a"}, b"audio", None)
    assert len(collection.inserted) == 1
    assert collection.inserted[0]["wer"] == 0.05
    assert "text" not in collection.inserted[0]
    assert "stratum" not in collection.inserted[0]


def test_persistence_failure_does_not_affect_observation():
    store = _FakeRecordStore(error=OSError("mongo down"))
    trends = TTSQualityStatistics()
    observer = TTSQualityObserver(
        TTSQualityGateway(_Evaluator()), trends, TTSRequestSampler(sample_rate=1.0),
        TTSAlertService(), executor=_InlineExecutor(), record_store=store,
    )
    observer.observe(_lightweight_result(), b"audio", None)
    assert trends.wer_statistics()["mean"] == 0.05


def test_repository_creates_time_series_collection_with_ttl_and_redacts_text():
    database = _FakeDatabase()
    clock = _MutableClock(9)
    repository = MongoTTSQualityRecordRepository(
        retention_days=7, clock=clock, client_factory=lambda uri: _FakeClient(database)
    )
    repository.save({
        "status": "pass", "text": "private", "stratum": "user-a",
        "audio": b"raw", "wer": 0.05, "first_chunk_ms": 123.0,
    })
    name, options = database.created[0]
    assert name == TTS_MONGO_COLLECTION_QUALITY
    assert options["timeseries"]["timeField"] == "recorded_at"
    assert options["expireAfterSeconds"] == 7 * 86400
    payload = database.collections[name].inserted[0]
    assert payload["recorded_at"] == clock()
    assert payload["wer"] == 0.05
    assert {"text", "stratum", "audio"}.isdisjoint(payload)


def test_repository_still_inserts_when_collection_already_exists():
    database = _FakeDatabase(create_error=RuntimeError("exists"))
    repository = MongoTTSQualityRecordRepository(
        clock=_MutableClock(9), client_factory=lambda uri: _FakeClient(database)
    )
    repository.save({"status": "pass"})
    assert len(database.collections[TTS_MONGO_COLLECTION_QUALITY].inserted) == 1


def test_repository_rejects_invalid_retention():
    with pytest.raises(ValueError):
        MongoTTSQualityRecordRepository(retention_days=0)
