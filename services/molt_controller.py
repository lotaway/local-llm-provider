from datetime import datetime, timezone
from typing import Dict, List, Any

from constants import LEARNING
from interfaces.molt_types import IMoltController


class MoltController(IMoltController):
    RECALL_HIT_RATE_THRESHOLD = 0.3
    MIN_ABSTRACTION_RATE = 0.01
    RECALL_STEP = 0.05
    DECAY_STEP = 0.005
    ABSTRACTION_STEP = 0.05
    RECALL_FLOOR = 0.3
    DECAY_CEILING = 0.05
    ABSTRACTION_FLOOR = 0.4
    LOW_IMPORTANCE_THRESHOLD = 0.4

    def __init__(
        self,
        mongo_repo=None,
        neo4j_repo=None,
        decay_scheduler=None,
        abstraction_engine=None,
        version_manager=None,
        feedback_service=None,
    ):
        self._mongo_repo = mongo_repo
        self._neo4j_repo = neo4j_repo
        self._decay_scheduler = decay_scheduler
        self._abstraction_engine = abstraction_engine
        self._version_manager = version_manager
        self._feedback_service = feedback_service
        self._recall_hit_rate = 0.5
        self._recall_threshold = 0.7
        self._decay_rate = 0.01
        self._abstraction_trigger_threshold = 0.6

    def run_decay_cycle(self) -> Dict[str, Any]:
        if self._decay_scheduler is None:
            return {"skipped": True, "reason": "decay_scheduler_unavailable"}
        return self._decay_scheduler.apply_decay_all()

    def run_abstraction_cycle(self) -> List[Dict[str, Any]]:
        if self._abstraction_engine is None:
            return []
        return self._abstraction_engine.auto_discover_and_abstract()

    def run_full_cycle(self) -> Dict[str, Any]:
        if not LEARNING:
            return {"skipped": True, "reason": "LEARNING disabled"}
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decay": self.run_decay_cycle(),
            "abstraction": self.run_abstraction_cycle(),
            "metrics": self.collect_metrics(),
        }

    def collect_metrics(self) -> Dict[str, Any]:
        episodes = self._episodic_memories()
        total = len(episodes)
        ltm_nodes = self._count_ltm_nodes()
        return {
            "total_memories": total,
            "avg_importance": self._average_importance(episodes),
            "recall_hit_rate": self._recall_hit_rate,
            "abstraction_rate": ltm_nodes / total if total else 0.0,
            "ltm_nodes": ltm_nodes,
        }

    def evaluate_and_adjust(self) -> Dict[str, Any]:
        metrics = self.collect_metrics()
        adjustments = self._collect_adjustments(metrics)
        return {
            "metrics": metrics,
            "adjustments": adjustments,
            "strategy": self.get_strategy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_strategy(self) -> Dict[str, float]:
        return {
            "recall_threshold": self._recall_threshold,
            "decay_rate": self._decay_rate,
            "abstraction_trigger_threshold": self._abstraction_trigger_threshold,
        }

    def _collect_adjustments(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        adjustments = []
        if metrics["recall_hit_rate"] < self.RECALL_HIT_RATE_THRESHOLD:
            self._recall_threshold = max(self.RECALL_FLOOR, self._recall_threshold - self.RECALL_STEP)
            adjustments.append(self._adjustment("recall_threshold", self._recall_threshold, "low_recall"))
        if metrics["avg_importance"] < self.LOW_IMPORTANCE_THRESHOLD:
            self._decay_rate = min(self.DECAY_CEILING, self._decay_rate + self.DECAY_STEP)
            adjustments.append(self._adjustment("decay_rate", self._decay_rate, "low_importance"))
        if metrics["abstraction_rate"] < self.MIN_ABSTRACTION_RATE:
            self._abstraction_trigger_threshold = max(
                self.ABSTRACTION_FLOOR,
                self._abstraction_trigger_threshold - self.ABSTRACTION_STEP,
            )
            adjustments.append(
                self._adjustment(
                    "abstraction_trigger_threshold",
                    self._abstraction_trigger_threshold,
                    "low_abstraction",
                )
            )
        return adjustments

    def _adjustment(self, target: str, value: float, reason: str) -> Dict[str, Any]:
        return {"type": target, "value": value, "reason": reason}

    def _episodic_memories(self) -> List[Dict[str, Any]]:
        if self._mongo_repo is None:
            return []
        return self._mongo_repo.get_by_type("episodic")

    def _average_importance(self, episodes: List[Dict[str, Any]]) -> float:
        if not episodes:
            return 0.0
        return sum(memory.get("importance_score", 0.0) for memory in episodes) / len(episodes)

    def _count_ltm_nodes(self) -> int:
        if self._neo4j_repo is None:
            return 0
        return self._neo4j_repo.count_ltm_nodes()
