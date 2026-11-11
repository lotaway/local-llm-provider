import hashlib
import io
from pathlib import Path

import pytest

from tts_estimate.config.settings import AppConfig
from tts_estimate.infrastructure.model_assets import (
    DNSMOS_ASSET,
    ModelAsset,
    ensure_local_model,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def _asset(payload: bytes) -> ModelAsset:
    return ModelAsset(
        name="fixture-model",
        url="https://example.invalid/model.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        version="fixture-1",
    )


def _opener(payload: bytes, calls: list):
    def open(url, timeout=None):
        calls.append(url)
        return io.BytesIO(payload)

    return open


def test_existing_file_with_pinned_checksum_is_reused_without_download(tmp_path):
    payload = b"model-bytes"
    target = tmp_path / "models" / "fixture-model.bin"
    target.parent.mkdir()
    target.write_bytes(payload)
    calls: list = []
    path = ensure_local_model(target, _asset(payload), opener=_opener(b"other", calls))
    assert path == target
    assert calls == []


def test_missing_file_is_downloaded_verified_and_installed(tmp_path):
    payload = b"model-bytes"
    target = tmp_path / "models" / "nested" / "fixture-model.bin"
    calls: list = []
    path = ensure_local_model(target, _asset(payload), opener=_opener(payload, calls))
    assert path.read_bytes() == payload
    assert calls == ["https://example.invalid/model.bin"]
    assert list(target.parent.glob("*.part")) == []


def test_download_with_wrong_checksum_is_rejected_and_leaves_no_file(tmp_path):
    target = tmp_path / "models" / "fixture-model.bin"
    with pytest.raises(ValueError):
        ensure_local_model(target, _asset(b"expected"), opener=_opener(b"corrupted", []))
    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []


def test_mismatched_existing_file_is_quarantined_and_replaced(tmp_path):
    target = tmp_path / "models" / "fixture-model.bin"
    target.parent.mkdir()
    target.write_bytes(b"outdated-content")
    payload = b"fresh-model-bytes"
    path = ensure_local_model(target, _asset(payload), opener=_opener(payload, []))
    assert path.read_bytes() == payload
    assert list(target.parent.glob("*.rejected-*")) == []


def test_failed_download_rolls_back_the_previous_file(tmp_path):
    target = tmp_path / "models" / "fixture-model.bin"
    target.parent.mkdir()
    target.write_bytes(b"previous-content")

    def failing(url, timeout=None):
        raise OSError("network down")

    with pytest.raises(OSError):
        ensure_local_model(target, _asset(b"new-content"), opener=failing)
    assert target.read_bytes() == b"previous-content"
    assert list(target.parent.glob("*.rejected-*")) == []


def test_file_sha256_returns_none_for_missing_files(tmp_path):
    assert file_sha256(tmp_path / "missing.bin") is None


def test_dnsmos_asset_pins_the_official_model_file():
    assert DNSMOS_ASSET.url.endswith("DNSMOS/DNSMOS/sig_bak_ovr.onnx")
    assert DNSMOS_ASSET.sha256 == "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"


def test_model_settings_support_nested_environment_overrides(monkeypatch):
    monkeypatch.setenv("TTS_EVAL_MODELS__WHISPER_MODEL", "small")
    monkeypatch.setenv("TTS_EVAL_MODELS__MODELS_ROOT", "artifacts/tts")
    monkeypatch.setenv("TTS_EVAL_MODELS__DNSMOS_MODEL_PATH", "artifacts/tts/dnsmos.onnx")
    config = AppConfig()
    assert config.models.whisper_model == "small"
    assert config.models.models_root == "artifacts/tts"
    assert config.models.dnsmos_model_path == "artifacts/tts/dnsmos.onnx"


def test_threshold_settings_support_nested_environment_overrides(monkeypatch):
    monkeypatch.setenv("TTS_EVAL_THRESHOLDS__WER_MAX", "0.2")
    monkeypatch.setenv("TTS_EVAL_THRESHOLDS__SIMILARITY_MIN", "0.5")
    config = AppConfig()
    assert config.thresholds.wer_max == 0.2
    assert config.thresholds.similarity_min == 0.5


def test_default_model_settings_are_unchanged():
    config = AppConfig()
    assert config.models.whisper_model == "large"
    assert config.models.dnsmos_model_path == "models/dnsmos.onnx"
    assert config.models.ecapa_model_source == "speechbrain/spkrec-ecapa-voxceleb"
    assert config.models.models_root == "models"


def _fake_components(monkeypatch, created):
    class FakeWhisper:
        def __init__(self, model_name="large", download_root=None):
            created["whisper"] = (model_name, download_root)

    class FakeDNSMOS:
        def __init__(self, model_path="", session=None):
            created["dnsmos"] = model_path

    class FakeECAPA:
        def __init__(self, model_source="", savedir=None):
            created["ecapa"] = (model_source, savedir)

    monkeypatch.setattr("tts_estimate.infrastructure.whisper_asr.WhisperASRProcessor", FakeWhisper)
    monkeypatch.setattr("tts_estimate.infrastructure.dnsmos.MicrosoftDNSMOSCalculator", FakeDNSMOS)
    monkeypatch.setattr(
        "tts_estimate.infrastructure.speaker_similarity.ECAPASpeakerSimilarityCalculator", FakeECAPA
    )
    return created


def test_factory_directs_every_model_into_the_project_models_directory(tmp_path, monkeypatch):
    created = _fake_components(monkeypatch, {})
    resolved = tmp_path / "dnsmos.onnx"
    monkeypatch.setattr(
        "tts_estimate.infrastructure.model_assets.ensure_local_model",
        lambda path, asset, opener=None: resolved,
    )
    monkeypatch.setenv("TTS_EVAL_MODELS__MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_EVAL_MODELS__DNSMOS_MODEL_PATH", str(resolved))
    from tts_estimate.infrastructure.evaluator_factory import build_quality_evaluator

    evaluator = build_quality_evaluator()
    assert created == {
        "whisper": ("large", str(tmp_path / "whisper")),
        "dnsmos": str(resolved),
        "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", str(tmp_path / "ecapa")),
    }
    assert evaluator is not None


def test_evaluator_stack_is_composed_only_by_the_shared_factory():
    adapter_source = (ROOT / "infrastructure" / "tts" / "quality_evaluator_adapter.py").read_text(
        encoding="utf-8"
    )
    cli_source = (ROOT / "tts_estimate" / "main.py").read_text(encoding="utf-8")
    factory_source = (
        ROOT / "tts_estimate" / "infrastructure" / "evaluator_factory.py"
    ).read_text(encoding="utf-8")
    assert "build_quality_evaluator" in adapter_source
    assert "build_quality_evaluator" in cli_source
    assert "WhisperASRProcessor(" not in adapter_source
    assert "WhisperASRProcessor(" not in cli_source
    assert "ensure_local_model" in factory_source
