import os
import tempfile
import threading
from typing import Any, Callable, Dict, Optional

from infrastructure.tts.wav_bytes import merge_wav_segments
from interfaces.tts_types import ITTSQualityEvaluator


def build_default_evaluator() -> Any:
    from tts_estimate.application.evaluator_service import TTSQualityEvaluator
    from tts_estimate.config.settings import AppConfig
    from tts_estimate.infrastructure.dnsmos import MicrosoftDNSMOSCalculator
    from tts_estimate.infrastructure.speaker_similarity import ECAPASpeakerSimilarityCalculator
    from tts_estimate.infrastructure.wer_calculator import JiwerWERCalculator
    from tts_estimate.infrastructure.whisper_asr import WhisperASRProcessor

    config = AppConfig()
    return TTSQualityEvaluator(
        WhisperASRProcessor(model_name=config.models.whisper_model),
        MicrosoftDNSMOSCalculator(model_path=config.models.dnsmos_model_path),
        ECAPASpeakerSimilarityCalculator(model_source=config.models.ecapa_model_source),
        JiwerWERCalculator(),
    )


class ProductionTTSQualityEvaluator(ITTSQualityEvaluator):
    def __init__(
        self,
        evaluator_factory: Optional[Callable[[], Any]] = None,
        temp_dir: Optional[str] = None,
    ) -> None:
        self._factory = evaluator_factory or build_default_evaluator
        self._temp_dir = temp_dir
        self._evaluator: Any = None
        self._lock = threading.Lock()

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
                    self._evaluator = self._factory()
        return self._evaluator

    @staticmethod
    def _write_audio(payload: bytes, workdir: str, filename: str) -> str:
        path = os.path.join(workdir, filename)
        with open(path, "wb") as handle:
            handle.write(merge_wav_segments(payload))
        return path
