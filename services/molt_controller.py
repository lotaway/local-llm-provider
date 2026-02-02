from datetime import datetime
from typing import Dict, List
from constants import LEARNING
from interfaces.molt_types import IMoltController
from interfaces.memory_types import IMemoryRepository
from interfaces.decay_types import IDecayCalculator


class MoltController(IMoltController):
    def __init__(self, memory_repo: IMemoryRepository, decay_calc: IDecayCalculator):
        self._repo = memory_repo
        self._decay_calc = decay_calc

    def run_decay_cycle(self) -> Dict:
        from services.decay_scheduler import DecayScheduler
        scheduler = DecayScheduler(self._repo, self._decay_calc)
        return scheduler.apply_decay_all()

    def run_abstraction_cycle(self) -> List[Dict]:
        from services.abstraction_engine import AbstractionEngine
        engine = AbstractionEngine(self._repo)
        return engine.auto_discover_and_abstract()

    def run_full_cycle(self) -> Dict:
        if not LEARNING:
            return {"skipped": True, "reason": "LEARNING disabled"}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "decay": self.run_decay_cycle(),
            "abstraction": self.run_abstraction_cycle(),
        }

    def collect_metrics(self) -> Dict:
        memories = self._repo.get_by_type("episodic")
        avg_importance = sum(m.get("importance_score", 0) for m in memories) / len(memories) if memories else 0
        return {
            "total_memories": len(memories),
            "avg_importance": avg_importance,
        }
