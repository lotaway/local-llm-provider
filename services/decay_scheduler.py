from constants import LEARNING
from interfaces.memory_types import IMemoryRepository
from interfaces.decay_types import IDecayCalculator


class DecayScheduler:
    DISCARD_THRESHOLD = 0.1

    def __init__(self, memory_repo: IMemoryRepository, decay_calculator: IDecayCalculator):
        self._repo = memory_repo
        self._calculator = decay_calculator

    def apply_decay_all(self) -> dict:
        if not LEARNING:
            return {"skipped": True}

        memories = self._repo.get_by_type("episodic")
        updated = 0

        for memory in memories:
            new_score = self._calculator.calculate(memory)
            self._repo.update_importance(memory["chunk_id"], new_score)
            updated += 1

        return {"updated": updated}

    def cleanup_low_importance(self) -> list:
        if not LEARNING:
            return []

        low_memories = self._repo.get_low_importance(self.DISCARD_THRESHOLD, 1000)
        discarded = []

        for memory in low_memories:
            self._repo.mark_discarded(memory["chunk_id"])
            discarded.append(memory["chunk_id"])

        return discarded
