import re
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_EMOTION_PARAMS = {
    "valence": 0.5,
    "arousal": 0.5,
    "dominance": 0.5,
    "speed": 0.5,
    "pitch": 0.5,
    "energy": 0.5,
}

SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]*")
POSITIVE_WORDS = frozenset(
    {
        "好",
        "great",
        "good",
        "excellent",
        "amazing",
        "love",
        "like",
        "best",
        "happy",
        "success",
        "win",
        "fast",
        "clear",
        "cool",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "poor",
        "fail",
        "failure",
        "slow",
        "broken",
        "wrong",
        "problem",
        "hate",
    }
)
MAX_TEXT_LENGTH = 1200


@dataclass(frozen=True)
class EmotionParams:
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    speed: float = 0.5
    pitch: float = 0.5
    energy: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "speed": self.speed,
            "pitch": self.pitch,
            "energy": self.energy,
        }


class EmotionParser:
    def __init__(self, default_params: Optional[Dict[str, float]] = None):
        params = default_params or DEFAULT_EMOTION_PARAMS
        self._default = EmotionParams(
            valence=float(params.get("valence", 0.5)),
            arousal=float(params.get("arousal", 0.5)),
            dominance=float(params.get("dominance", 0.5)),
            speed=float(params.get("speed", 0.5)),
            pitch=float(params.get("pitch", 0.5)),
            energy=float(params.get("energy", 0.5)),
        )

    def parse(self, text: str) -> EmotionParams:
        sentences = [item.strip() for item in SENTENCE_PATTERN.findall(text or "") if item.strip()]
        valence = self._valence(text or "")
        arousal = self._arousal(text or "", sentences)
        dominance = self._dominance(sentences)
        speed = self._speed(sentences)
        return EmotionParams(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            speed=speed,
            pitch=0.5 + valence * 0.5,
            energy=arousal,
        )

    def parse_params(self, text: str) -> Dict[str, float]:
        return self.parse(text).to_dict()

    def _valence(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())[:64]
        if not tokens:
            return self._default.valence
        scores = [
            1.0 if token in POSITIVE_WORDS else -1.0 if token in NEGATIVE_WORDS else 0.0
            for token in tokens
        ]
        polarity = sum(scores) / len(scores) if scores else 0.0
        return max(0.0, min(1.0, (polarity + 1.0) / 2.0))

    def _arousal(self, text: str, sentences: List[str]) -> float:
        if not text.strip():
            return self._default.arousal
        intensity = min(text.count("!"), 3) * 0.2
        intensity += min(text.count("?"), 2) * 0.15
        intensity += min(len(text) / MAX_TEXT_LENGTH, 1.0) * 0.05
        return max(0.0, min(1.0, 0.3 + intensity))

    def _dominance(self, sentences: List[str]) -> float:
        if not sentences:
            return self._default.dominance
        questions = sum(1 for sentence in sentences if sentence.rstrip().endswith("?"))
        question_ratio = questions / len(sentences)
        statement_dominance = 0.9
        default_dominance = self._default.dominance
        return max(
            0.0,
            min(1.0, statement_dominance - question_ratio * (statement_dominance - default_dominance)),
        )

    def _speed(self, sentences: List[str]) -> float:
        if not sentences:
            return self._default.speed
        average_length = sum(len(sentence) for sentence in sentences) / len(sentences)
        return max(0.0, min(1.0, (14.0 / max(1.0, average_length)) / 20.0))
