from datetime import datetime
from typing import Dict, List, Any, Optional
from constants import LEARNING
from interfaces.molt_types import IMoltController
from interfaces.memory_types import IMemoryRepository
from interfaces.decay_types import IDecayCalculator


class MoltController(IMoltController):
    RECALL_HIT_RATE_THRESHOLD = 0.3
    MIN_ABSTRACTION_RATE = 0.01

    def __init__(
        self,
        memory_repo: IMemoryRepository = None,
        decay_calc: IDecayCalculator = None,
        mongo_repo=None,
        neo4j_repo=None,
        feedback_service=None,
        decay_scheduler=None,
        abstraction_engine=None,
        version_manager=None,
    ):
        self._repo = memory_repo or mongo_repo
        self._decay_calc = decay_calc
        self._neo4j_repo = neo4j_repo
        self._feedback_service = feedback_service
        self._decay_scheduler = decay_scheduler
        self._abstraction_engine = abstraction_engine
        self._version_manager = version_manager

        # strategy params
        self._recall_threshold = 0.7
        self._decay_rate = 0.01
        self._abstraction_trigger_threshold = 0.6

    def run_decay_cycle(self) -> Dict:
        if self._decay_scheduler and hasattr(self._decay_scheduler, "apply_decay_all"):
            return self._decay_scheduler.apply_decay_all()
        if not self._repo or not self._decay_calc:
            return {"skipped": True, "reason": "not_configured"}
        from services.decay_scheduler import DecayScheduler

        scheduler = DecayScheduler(self._repo, self._decay_calc)
        return scheduler.apply_decay_all()

    def run_abstraction_cycle(self) -> List[Dict]:
        if self._abstraction_engine and hasattr(self._abstraction_engine, "auto_discover_and_abstract"):
            return self._abstraction_engine.auto_discover_and_abstract()
        if not self._repo:
            return []
        from services.abstraction_engine import AbstractionEngine

        engine = AbstractionEngine(self._repo)
        return engine.auto_discover_and_abstract()

    def run_full_cycle(self) -> Dict:
        if not LEARNING:
            return {"skipped": True, "reason": "LEARNING disabled"}

        decay_res = self.run_decay_cycle()
        abs_res = self.run_abstraction_cycle()
        metrics = self.collect_metrics()

        # normalize keys for compatibility
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "decay": decay_res,
            "abstraction": abs_res,
            "metrics": metrics,
            "decay_stats": decay_res,
            "abstraction_results": abs_res,
        }
        return result

    def collect_metrics(self) -> Dict:
        if self._repo and hasattr(self._repo, "get_by_type"):
            memories = self._repo.get_by_type("episodic")
            avg_importance = sum(m.get("importance_score", 0) for m in memories) / len(memories) if memories else 0
            return {
                "total_memories": len(memories),
                "avg_importance": avg_importance,
                "recall_hit_rate": getattr(self, "_last_recall_hit_rate", 0.5),
                "stale_chunks": 0,
                "abstraction_rate": 0.01,
                "total_chunks": len(memories),
                "episodic_chunks": len(memories),
                "ltm_nodes": 0,
                "discarded_chunks": 0,
            }

        # try alternative repos
        total = 0
        avg_imp = 0.5
        if self._repo and hasattr(self._repo, "chunks"):
            try:
                total = self._repo.chunks.count_documents({}) if hasattr(self._repo.chunks, "count_documents") else total
            except Exception:
                pass
        if self._neo4j_repo and hasattr(self._neo4j_repo, "get_ltm_by_topic"):
            pass
        return {
            "total_memories": total,
            "avg_importance": avg_imp,
            "recall_hit_rate": 0.5,
            "stale_chunks": 0,
            "abstraction_rate": 0.01,
            "total_chunks": total,
            "episodic_chunks": total,
            "ltm_nodes": 0,
            "discarded_chunks": 0,
        }

    def evaluate_and_adjust(self) -> Dict[str, Any]:
        metrics = self.collect_metrics()
        adjustments = []

        recall_hit_rate = metrics.get("recall_hit_rate", 0.5)
        if recall_hit_rate < self.RECALL_HIT_RATE_THRESHOLD:
            self._recall_threshold = max(0.3, self._recall_threshold - 0.05)
            adjustments.append({"type": "recall_threshold", "value": self._recall_threshold, "reason": "low_recall"})

        avg_importance = metrics.get("avg_importance", 0.5)
        if avg_importance < 0.4:
            self._decay_rate = min(0.05, self._decay_rate + 0.005)
            adjustments.append({"type": "decay_rate", "value": self._decay_rate, "reason": "low_importance"})

        abstraction_rate = metrics.get("abstraction_rate", 0.01)
        if abstraction_rate < self.MIN_ABSTRACTION_RATE:
            self._abstraction_trigger_threshold = max(0.4, self._abstraction_trigger_threshold - 0.05)
            adjustments.append({"type": "abstraction_trigger_threshold", "value": self._abstraction_trigger_threshold, "reason": "low_abstraction"})

        return {
            "metrics": metrics,
            "adjustments": adjustments,
            "strategy": {
                "recall_threshold": self._recall_threshold,
                "decay_rate": self._decay_rate,
                "abstraction_trigger_threshold": self._abstraction_trigger_threshold,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def get_strategy(self) -> Dict[str, float]:
        return {
            "recall_threshold": self._recall_threshold,
            "decay_rate": self._decay_rate,
            "abstraction_trigger_threshold": self._abstraction_trigger_threshold,
        }
