import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_styletts2_submodule_has_vendor_sources():
    vendor = ROOT / "third_party" / "StyleTTS2"
    assert (vendor / "models.py").is_file()
    assert (vendor / "utils.py").is_file()
    assert (vendor / "requirements.txt").is_file()


def test_tts_dependencies_live_only_in_pyproject():
    import tomllib

    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)

    tts_dependencies = project["project"]["optional-dependencies"]["tts"]
    joined = "\n".join(tts_dependencies)

    for dependency in (
        "phonemizer",
        "espeakng-loader",
        "monotonic-align",
        "vector-quantize-pytorch",
    ):
        assert dependency in joined

    # torch/torchaudio must come from the host-specific ROCm/CUDA/CPU build.
    assert not any(item.startswith(("torch>=", "torchaudio>=")) for item in tts_dependencies)
    assert not (ROOT / "requirements-tts.txt").exists()


def test_setup_script_does_not_install_torch():
    script = (ROOT / "scripts" / "setup_styletts2.sh").read_text(encoding="utf-8")

    assert "requirements-tts.txt" not in script
    assert "host build preserved" in script
    assert "optional-dependencies" in script
    assert "--constraint" in script


def test_setup_scripts_are_executable():
    for relative in ("scripts/setup_styletts2.sh", "scripts/check_styletts2.py"):
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_mode & stat.S_IXUSR


def test_runtime_report_marks_missing_checkpoint_as_optional(tmp_path):
    from utils.tts_runtime import styletts2_dependency_report

    report = styletts2_dependency_report(ROOT)
    assert report["vendor_present"] is True
    assert report["vendor_path"] == str((ROOT / "third_party" / "StyleTTS2").resolve())
    assert "checkpoint" not in report


def test_adapter_accepts_vendor_cloner_hook():
    pytest.importorskip("torch")
    pytest.importorskip("redis")

    from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter
    from services.emotion_parser import EmotionParser

    class FakeRedis:
        def get(self, key):
            return None

        def set(self, key, value):
            return None

    async def fake_cloner(audio_data, alias):
        return {"cloned": True, "alias": alias, "size": len(audio_data)}

    adapter = StyleTTS2Adapter(
        redis_client=FakeRedis(),
        emotion_parser=EmotionParser(),
        vendor_cloner=fake_cloner,
    )

    import asyncio

    result = asyncio.run(adapter.clone_voice(b"audio", user_id="u1", alias="voice-a"))
    assert result["status"] == "ok"
    assert result["result"]["cloned"] is True
    assert result["result"]["size"] == 5
