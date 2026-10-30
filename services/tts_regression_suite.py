import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from constants import TTS_AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


DEFAULT_TEST_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Please call back tomorrow at nine in the morning.",
]


class TTSRegressionSuite:
    def __init__(self, sentences: Optional[List[str]] = None):
        self.sentences = list(sentences) if sentences is not None else list(DEFAULT_TEST_SENTENCES)
        self._runs: List[Dict[str, Any]] = []

    def run(self, generate: Callable[[str], bytes]) -> Dict[str, Any]:
        results = []
        for sentence in self.sentences:
            started_at = time.monotonic()
            audio = generate(sentence)
            results.append(
                {
                    "sentence": sentence,
                    "version": str(uuid.uuid4())[:8],
                    "duration_seconds": len(audio) / (TTS_AUDIO_SAMPLE_RATE * 4),
                    "latency_ms": round((time.monotonic() - started_at) * 1000, 3),
                }
            )
        run_record = {"version": str(uuid.uuid4())[:8], "results": results}
        self._runs.append(run_record)
        return run_record

    def compare(self, current: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if previous is None:
            return {"compared": False, "changed": 0, "sentences": []}
        current_scores = {item["sentence"]: item for item in current.get("results", [])}
        previous_scores = {item["sentence"]: item for item in previous.get("results", [])}
        changed = sum(
            1
            for sentence in self.sentences
            if current_scores.get(sentence, {}).get("version")
            != previous_scores.get(sentence, {}).get("version")
        )
        return {"compared": True, "changed": changed, "sentences": list(self.sentences)}
