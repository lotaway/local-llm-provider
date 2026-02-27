from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class AudioQualityMetrics:
    wer: float
    dnsmos_overall: float
    dnsmos_signal: float
    dnsmos_background: float
    speaker_similarity: float

@dataclass(frozen=True)
class EvaluationResult:
    metrics: AudioQualityMetrics
    is_passed: bool

class ASRProcessor(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

class MOSCalculator(ABC):
    @abstractmethod
    def calculate_score(self, audio_path: str) -> Dict[str, float]:
        pass

class SpeakerSimilarityCalculator(ABC):
    @abstractmethod
    def calculate_similarity(self, reference_audio: str, generated_audio: str) -> float:
        pass
