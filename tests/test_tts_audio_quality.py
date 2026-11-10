import io
import wave

import pytest

from services.tts_audio_quality import TTSAudioQualityService


def _wav(frames: bytes, rate: int = 24000, width: int = 2, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def _pcm16(seconds: float, rate: int = 24000) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


def test_duration_of_a_pcm16_segment_matches_the_container_format():
    service = TTSAudioQualityService(sample_rate=24000)

    assert service.duration_seconds(_wav(_pcm16(3.0))) == pytest.approx(3.0)


def test_streamed_segments_sum_their_durations():
    service = TTSAudioQualityService(sample_rate=24000)
    payload = _wav(_pcm16(0.5)) + _wav(_pcm16(0.25))

    assert service.duration_seconds(payload) == pytest.approx(0.75)


def test_headerless_payload_falls_back_to_pcm16_samples():
    service = TTSAudioQualityService(sample_rate=8000)

    assert service.duration_seconds(b"\x00" * 16000) == pytest.approx(1.0)


def test_duration_anomaly_compares_against_the_configured_limit():
    service = TTSAudioQualityService(sample_rate=24000, max_duration_seconds=2.0)

    assert service.is_duration_anomaly(_wav(_pcm16(1.0))) is False
    assert service.is_duration_anomaly(_wav(_pcm16(3.0))) is True
    assert service.is_duration_anomaly(_wav(b"")) is True


def test_duplicate_detection_keeps_only_digests():
    service = TTSAudioQualityService()

    first = _wav(_pcm16(0.1))
    second = _wav(_pcm16(0.1))

    assert service.has_duplicate(first) is False
    assert service.has_duplicate(second) is True
    assert len(service._history) == 1
    assert isinstance(service._history[0], str)
    assert len(service._history[0]) == 64


def test_duplicate_history_is_bounded():
    service = TTSAudioQualityService(history_limit=2)

    for seconds in (0.1, 0.2, 0.3):
        assert service.has_duplicate(_wav(_pcm16(seconds))) is False

    assert len(service._history) == 2


def test_duplicate_history_limit_is_validated():
    with pytest.raises(ValueError):
        TTSAudioQualityService(history_limit=0)
