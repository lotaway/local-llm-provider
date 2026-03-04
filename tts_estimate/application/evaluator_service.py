from ..domain.interfaces import (
    ASRProcessor, MOSCalculator, SpeakerSimilarityCalculator, 
    AudioQualityMetrics, EvaluationResult
)
from ..infrastructure.wer_calculator import TextComparison

class TTSQualityEvaluator:
    def __init__(
        self, 
        asr: ASRProcessor, 
        mos: MOSCalculator, 
        speaker: SpeakerSimilarityCalculator,
        wer_calc: TextComparison
    ):
        self._asr = asr
        self._mos = mos
        self._speaker = speaker
        self._wer_calc = wer_calc

    def evaluate(
        self, 
        reference_text: str, 
        reference_audio: str, 
        generated_audio: str,
        thresholds: dict
    ) -> EvaluationResult:
        metrics = self._collect_metrics(
            reference_text, reference_audio, generated_audio
        )
        is_passed = self._check_quality(metrics, thresholds)
        return EvaluationResult(metrics=metrics, is_passed=is_passed)

    def _collect_metrics(
        self, ref_text: str, ref_audio: str, gen_audio: str
    ) -> AudioQualityMetrics:
        transcription = self._asr.transcribe(gen_audio)
        wer_val = self._wer_calc.compute_wer(ref_text, transcription)
        mos_scores = self._mos.calculate_score(gen_audio)
        similarity = self._speaker.calculate_similarity(ref_audio, gen_audio)
        
        return AudioQualityMetrics(
            wer=wer_val,
            dnsmos_overall=mos_scores["overall"],
            dnsmos_signal=mos_scores["signal"],
            dnsmos_background=mos_scores["background"],
            speaker_similarity=similarity
        )

    def _check_quality(self, metrics: AudioQualityMetrics, thresholds: dict) -> bool:
        if metrics.wer > thresholds["wer_max"]:
            return False
        if metrics.dnsmos_overall < thresholds["dnsmos_min"]:
            return False
        if metrics.speaker_similarity < thresholds["similarity_min"]:
            return False
        return True
