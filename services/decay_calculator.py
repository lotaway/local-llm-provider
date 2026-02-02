import math
from datetime import datetime
from typing import Dict, Any
from interfaces.decay_types import IDecayCalculator


class ExponentialDecayCalculator(IDecayCalculator):
    DEFAULT_DECAY_RATE = 0.01
    REINFORCEMENT_FACTOR = 0.05

    def calculate(self, memory: Dict[str, Any]) -> float:
        current = memory.get("importance_score", 0.5)
        last_reviewed = memory.get("last_reviewed") or memory.get("created_at")
        decay_rate = memory.get("decay_rate", self.DEFAULT_DECAY_RATE)
        review_count = memory.get("review_count", 0)

        if isinstance(last_reviewed, str):
            last_reviewed = datetime.fromisoformat(last_reviewed.replace("Z", "+00:00"))

        days_passed = (datetime.utcnow() - last_reviewed).total_seconds() / 86400
        decayed = current * math.exp(-decay_rate * days_passed)
        reinforced = decayed * (1 + review_count * self.REINFORCEMENT_FACTOR)

        return max(0.0, min(1.0, reinforced))
