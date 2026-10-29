import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TTSSamplingMonitor:
    def __init__(self, sample_rate: float = 0.05):
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be within (0, 1], got {sample_rate}")
        self._sample_rate = sample_rate
        self._sampled = 0
        self._results: List[Dict[str, Any]] = []

    def should_sample(self) -> bool:
        sample = random.random() < self._sample_rate
        if sample:
            self._sampled += 1
        return sample

    def record(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self._results.append(dict(result))
        return dict(result)

    def results(self, limit: int = 1000) -> List[Dict[str, Any]]:
        return list(self._results[:limit])

    @property
    def sampled_count(self) -> int:
        return self._sampled
