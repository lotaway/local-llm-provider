import time

from services.tts_audio_quality import TTSAudioQualityService


def test_audio_quality_detects_duration_anomaly():
    service = TTSAudioQualityService(sample_rate=24000, max_duration_seconds=2.0)

    assert service.is_duration_anomaly(b"\x00\x00\x00\x00") is False
    assert service.is_duration_anomaly(b"\x00" * (24000 * 3 * 4)) is True


def test_audio_quality_detects_duplicate_fragment():
    service = TTSAudioQualityService()

    first = b"\x00\x01\x00\x02"
    second = b"\x00\x01\x00\x02"

    assert service.has_duplicate(first) is False
    assert service.has_duplicate(second) is True


def test_audio_quality_duration_seconds_uses_sample_rate():
    service = TTSAudioQualityService(sample_rate=8000)

    assert abs(service.duration_seconds(b"\x00" * 16000) - 0.5) < 1e-9
