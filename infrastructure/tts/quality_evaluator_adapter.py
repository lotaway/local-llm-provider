import os
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional

from constants import TTS_EVALUATOR_RETRY_COOLDOWN_SECONDS
from infrastructure.tts.wav_bytes import merge_wav_segments
from interfaces.tts_types import ITTSQualityEvaluator


def build_default_evaluator() -> Any:
    from tts_estimate.infrastructure.evaluator_factory import build_quality_evaluator

    return build_quality_evaluator()


class ProductionTTSQualityEvaluator(ITTSQualityEvaluator):
    def __init__(
        self,
        evaluator_factory: Optional[Callable[[], Any]] = None,
        temp_dir: Optional[str] = None,
        retry_cooldown_seconds: float = TTS_EVALUATOR_RETRY_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = evaluator_factory or build_default_evaluator
        self._temp_dir = temp_dir
        self._evaluator: Any = None
        self._lock = threading.Lock()
        self._cooldown = retry_cooldown_seconds
        self._clock = clock
        self._last_failure: Optional[tuple[float, BaseException]] = None

    def evaluate(
        self,
        ref_text: str,
        ref_audio: bytes,
        gen_audio: bytes,
        thresholds: Dict[str, float],
    ) -> Any:
        evaluator = self._evaluator_instance()
        with tempfile.TemporaryDirectory(dir=self._temp_dir) as workdir:
            generated = self._write_audio(gen_audio, workdir, "generated.wav")
            reference = self._write_audio(ref_audio, workdir, "reference.wav") if ref_audio else ""
            return evaluator.evaluate(ref_text, reference, generated, thresholds)

    def _evaluator_instance(self) -> Any:
        if self._evaluator is None:
            with self._lock:
                if self._evaluator is None:
                    self._evaluator = self._construct()
        return self._evaluator

    def _construct(self) -> Any:
        if self._cooldown_active():
            _, error = self._last_failure
            raise RuntimeError("quality evaluator construction is in retry cooldown") from error
        try:
            evaluator = self._factory()
        except Exception as error:
            self._last_failure = (self._clock(), error)
            raise
        self._last_failure = None
        return evaluator

    def _cooldown_active(self) -> bool:
        if self._last_failure is None:
            return False
        failed_at, _ = self._last_failure
        return self._clock() - failed_at < self._cooldown

    @staticmethod
    def _write_audio(payload: bytes, workdir: str, filename: str) -> str:
        path = os.path.join(workdir, filename)
        with open(path, "wb") as handle:
            handle.write(merge_wav_segments(payload))
        return path
