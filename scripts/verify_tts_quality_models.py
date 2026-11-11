#!/usr/bin/env python3
"""Run one real-model TTS quality evaluation through the service pipeline.

The evaluation models are resolved on demand into the project models
directory, a real speech clip is replayed through TTSService.stream and the
sampled observation must produce WER, DNSMOS and speaker similarity values.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import AsyncGenerator, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.tts.quality_evaluator_adapter import ProductionTTSQualityEvaluator  # noqa: E402
from services.tts_alerts import TTSAlertService  # noqa: E402
from services.tts_quality_gateway import TTSQualityGateway  # noqa: E402
from services.tts_quality_observer import TTSQualityObserver  # noqa: E402
from services.tts_quality_statistics import TTSQualityStatistics  # noqa: E402
from services.tts_request_sampler import TTSRequestSampler  # noqa: E402
from services.tts_service import TTSService  # noqa: E402

FIXTURE_URL = "https://github.com/librosa/librosa-data/raw/main/audio/5703-47212-0000.ogg"
FIXTURE_TEXT = (
    "with her white paint and her scarlet smokestack the inverashiel one of the "
    "two small steamers that during the summer months plied up and down the loch "
    "and incidentally carried on communication between inverashiel and cryonon"
)
WAIT_TIMEOUT_SECONDS = 900.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_tts_quality_models")


class _ReplayAdapter:
    def __init__(self, audio: bytes) -> None:
        self._audio = audio

    async def generate_speech_stream(
        self, text, emotion_params, speaker_id, callback=None, user_id=""
    ) -> AsyncGenerator[str, None]:
        if callback is not None:
            callback({"type": "first_chunk", "elapsed_ms": 1.0})
        for start in range(0, len(self._audio), 4096):
            yield base64.b64encode(self._audio[start : start + 4096]).decode("ascii")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify TTS quality evaluation models")
    parser.add_argument("--audio", default=None, help="local real-speech audio file to replay")
    parser.add_argument("--text", default=None, help="transcript used as the WER reference")
    return parser.parse_args(argv)


def load_fixture(args: argparse.Namespace) -> bytes:
    if args.audio:
        return _to_sixteen_khz_wav(Path(args.audio).read_bytes())
    with urllib.request.urlopen(FIXTURE_URL, timeout=120) as response:
        payload = response.read()
    return _to_sixteen_khz_wav(payload)


def _to_sixteen_khz_wav(payload: bytes) -> bytes:
    import librosa

    audio, _ = librosa.load(io.BytesIO(payload), sr=16000, mono=True)
    return _pcm_wav_bytes(audio)


def _pcm_wav_bytes(audio) -> bytes:
    frames = (audio * 32768.0).clip(-32768, 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    return buffer.getvalue()


def build_service(audio: bytes) -> TTSService:
    trends = TTSQualityStatistics()
    sampler = TTSRequestSampler(sample_rate=1.0)
    alerts = TTSAlertService()
    observer = TTSQualityObserver(
        TTSQualityGateway(ProductionTTSQualityEvaluator()),
        trends,
        sampler,
        alerts,
    )
    return TTSService(
        adapter=_ReplayAdapter(audio),
        quality_observer=observer,
        trends=trends,
        sampling_monitor=sampler,
        alerts=alerts,
    )


async def _collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


def run_request(service: TTSService, text: str, audio: bytes) -> bytes:
    return asyncio.run(
        _collect(service.stream(text, user_id="verify", reference_audio=audio))
    )


def wait_for_evaluation(trends: TTSQualityStatistics) -> bool:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if trends.wer_statistics()["count"] > 0:
            return True
        time.sleep(0.5)
    return False


def report_metrics(trends: TTSQualityStatistics) -> bool:
    wer = trends.wer_statistics()
    dnsmos = trends.dnsmos_statistics()
    similarity = trends.similarity_statistics()
    logger.info(
        "verification_metrics wer=%.4f dnsmos_overall=%.4f similarity=%.4f counts=%s/%s/%s",
        wer["mean"],
        dnsmos["mean"],
        similarity["mean"],
        wer["count"],
        dnsmos["count"],
        similarity["count"],
    )
    missing = _missing_metrics(wer, dnsmos, similarity)
    if missing:
        logger.error("verification_incomplete", extra={"missing": missing})
        return False
    logger.info("verification_complete")
    return True


def _missing_metrics(*statistics: Dict[str, float]) -> List[str]:
    names = ("wer", "dnsmos", "similarity")
    return [
        name for name, stats in zip(names, statistics) if stats["count"] <= 0
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    text = args.text or FIXTURE_TEXT
    audio = load_fixture(args)
    service = build_service(audio)
    replayed = run_request(service, text, audio)
    if replayed != audio:
        logger.error("fixture_replay_mismatch")
        return 1
    if not wait_for_evaluation(service.trends):
        logger.error("evaluation_did_not_complete")
        return 1
    return 0 if report_metrics(service.trends) else 1


if __name__ == "__main__":
    raise SystemExit(main())
