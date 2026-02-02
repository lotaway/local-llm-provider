from typing import List, Dict, Optional
from constants import LEARNING
from interfaces.memory_types import IMemoryRepository
from interfaces.abstraction_types import IAbstractionEngine


class AbstractionEngine(IAbstractionEngine):
    MIN_EPISODES = 5
    MIN_AVG_IMPORTANCE = 0.6

    def __init__(self, memory_repo: IMemoryRepository):
        self._repo = memory_repo

    def can_abstraction_trigger(self, topic: str) -> bool:
        episodes, avg_importance = self._find_related(topic)
        return len(episodes) >= self.MIN_EPISODES and avg_importance >= self.MIN_AVG_IMPORTANCE

    def abstract_to_ltm(self, topic: str) -> Optional[Dict]:
        if not LEARNING:
            return None

        episodes, avg_importance = self._find_related(topic)
        if not self._can_trigger(len(episodes), avg_importance):
            return None

        return {"topic": topic, "episodes_count": len(episodes), "avg_importance": avg_importance}

    def auto_discover_and_abstract(self) -> List[Dict]:
        if not LEARNING:
            return []

        results = []
        episodes = self._repo.get_by_type("episodic")
        topics = {e["text"].split()[0] for e in episodes if e.get("text")}

        for topic in list(topics)[:10]:
            if self.can_abstraction_trigger(topic):
                result = self.abstract_to_ltm(topic)
                if result:
                    results.append(result)

        return results

    def _find_related(self, topic: str) -> tuple:
        episodes = [e for e in self._repo.get_by_type("episodic") 
                    if topic.lower() in e.get("text", "").lower()]
        avg = sum(e.get("importance_score", 0) for e in episodes) / len(episodes) if episodes else 0
        return episodes, avg

    def _can_trigger(self, count: int, avg_importance: float) -> bool:
        return count >= self.MIN_EPISODES and avg_importance >= self.MIN_AVG_IMPORTANCE
