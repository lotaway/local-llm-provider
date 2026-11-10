import hashlib
import struct
from typing import Iterable, List, Optional, Tuple

RIFF_MARKER = b"RIFF"
WAVE_MARKER = b"WAVE"
FORMAT_CHUNK = b"fmt "
DATA_CHUNK = b"data"
RIFF_HEADER_SIZE = 8
RIFF_MIN_SIZE = 12
CHUNK_HEADER_SIZE = 8
FORMAT_MIN_SIZE = 16
CHANNELS_OFFSET = 2
SAMPLE_RATE_OFFSET = 4
BITS_OFFSET = 14
PCM16_BYTES_PER_SAMPLE = 2


def duration_seconds(audio: bytes, sample_rate: int) -> float:
    segments = _wav_segments(audio)
    if not segments:
        return len(audio) / (sample_rate * PCM16_BYTES_PER_SAMPLE)
    return sum(segments)


def is_duration_anomaly(audio: bytes, sample_rate: int, max_duration_seconds: float) -> bool:
    duration = duration_seconds(audio, sample_rate)
    return duration <= 0.0 or duration > max_duration_seconds


def digest(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()


def is_duplicate(fingerprint: str, history: Iterable[str]) -> bool:
    return fingerprint in history


def _wav_segments(audio: bytes) -> List[float]:
    segments: List[float] = []
    cursor = 0
    while audio[cursor : cursor + 4] == RIFF_MARKER:
        segment = _read_segment(audio, cursor)
        if segment is None:
            return segments
        seconds, cursor = segment
        segments.append(seconds)
    return segments


def _read_segment(audio: bytes, start: int) -> Optional[Tuple[float, int]]:
    if audio[start + 8 : start + 12] != WAVE_MARKER:
        return None
    declared = struct.unpack_from("<I", audio, start + 4)[0]
    end = min(len(audio), start + RIFF_HEADER_SIZE + declared)
    if end < start + RIFF_MIN_SIZE:
        return None
    byte_rate, data_bytes = _read_chunks(audio, start + RIFF_MIN_SIZE, end)
    if byte_rate <= 0:
        return None
    return data_bytes / byte_rate, max(end, start + RIFF_MIN_SIZE)


def _read_chunks(audio: bytes, start: int, end: int) -> Tuple[int, int]:
    byte_rate = 0
    data_bytes = 0
    cursor = start
    while cursor + CHUNK_HEADER_SIZE <= end:
        chunk_id = audio[cursor : cursor + 4]
        size = struct.unpack_from("<I", audio, cursor + 4)[0]
        payload = cursor + CHUNK_HEADER_SIZE
        if chunk_id == FORMAT_CHUNK:
            byte_rate = _byte_rate(audio, payload, end)
        elif chunk_id == DATA_CHUNK:
            data_bytes = min(size, max(0, end - payload))
        cursor = payload + size + size % 2
    return byte_rate, data_bytes


def _byte_rate(audio: bytes, payload: int, end: int) -> int:
    if payload + FORMAT_MIN_SIZE > end:
        return 0
    channels = struct.unpack_from("<H", audio, payload + CHANNELS_OFFSET)[0]
    rate = struct.unpack_from("<I", audio, payload + SAMPLE_RATE_OFFSET)[0]
    bits = struct.unpack_from("<H", audio, payload + BITS_OFFSET)[0]
    return rate * channels * max(bits // 8, 1)
