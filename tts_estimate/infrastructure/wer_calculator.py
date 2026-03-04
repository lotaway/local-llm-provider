from jiwer import wer
from ..domain.interfaces import ABC, abstractmethod

class TextComparison(ABC):
    @abstractmethod
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        pass

class JiwerWERCalculator(TextComparison):
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        return float(wer(reference, hypothesis))
