import io
import struct
import wave
from typing import Any, List, Tuple

RIFF_MARKER = b"RIFF"
WAVE_MARKER = b"WAVE"
RIFF_HEADER_SIZE = 8
RIFF_MIN_SIZE = 12


def merge_wav_segments(payload: bytes) -> bytes:
    segments = _segments(payload)
    if not segments:
        raise ValueError("audio payload contains no WAV segment")
    frames, params = _collect(segments)
    return _write(params, frames)


def _segments(payload: bytes) -> List[bytes]:
    segments: List[bytes] = []
    cursor = 0
    while cursor + RIFF_MIN_SIZE <= len(payload):
        start = payload.find(RIFF_MARKER, cursor)
        if start < 0 or start + RIFF_MIN_SIZE > len(payload):
            return segments
        if payload[start + 8 : start + 12] != WAVE_MARKER:
            cursor = start + 4
            continue
        size = struct.unpack_from("<I", payload, start + 4)[0]
        end = min(start + RIFF_HEADER_SIZE + size, len(payload))
        segments.append(payload[start:end])
        cursor = max(end, start + RIFF_MIN_SIZE)
    return segments


def _collect(segments: List[bytes]) -> Tuple[bytes, Any]:
    params = None
    frames: List[bytes] = []
    for segment in segments:
        with wave.open(io.BytesIO(segment), "rb") as handle:
            current = handle.getparams()
            if params is not None and _format(current) != _format(params):
                raise ValueError("WAV segments must share an identical format")
            params = current
            frames.append(handle.readframes(handle.getnframes()))
    return b"".join(frames), params


def _write(params: Any, frames: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(frames)
    return buffer.getvalue()


def _format(params: Any) -> Tuple[Any, ...]:
    return (
        params.nchannels,
        params.sampwidth,
        params.framerate,
        params.comptype,
        params.compname,
    )
