import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from constants import (
    LEARNING,
    MOLT_STRATEGY_ABSTRACTION_FLOOR,
    MOLT_STRATEGY_ABSTRACTION_STEP,
    MOLT_STRATEGY_DECAY_CEILING,
    MOLT_STRATEGY_DECAY_STEP,
    MOLT_STRATEGY_HISTORY_LIMIT,
    MOLT_STRATEGY_INITIAL_ABSTRACTION_THRESHOLD,
    MOLT_STRATEGY_INITIAL_DECAY_RATE,
    MOLT_STRATEGY_INITIAL_RECALL_THRESHOLD,
    MOLT_STRATEGY_LOW_IMPORTANCE_THRESHOLD,
    MOLT_STRATEGY_MIN_ABSTRACTION_RATE,
    MOLT_STRATEGY_RECALL_FLOOR,
    MOLT_STRATEGY_RECALL_HIT_RATE_THRESHOLD,
    MOLT_STRATEGY_RECALL_STEP,
)
from interfaces.memory_types import ISensoryCache, IWorkingMemory
from interfaces.molt_types import (
    IMoltController,
    IStrategyHistoryStore,
    MoltStrategyConfig,
)
from services.strategy_history import InMemoryStrategyHistoryStore

logger = logging.getLogger(__name__)

RECENT_HISTORY_SIZE = 5


def default_strategy_config() -> MoltStrategyConfig:
    return MoltStrategyConfig(
        recall_hit_rate_threshold=MOLT_STRATEGY_RECALL_HIT_RATE_THRESHOLD,
        min_abstraction_rate=MOLT_STRATEGY_MIN_ABSTRACTION_RATE,
        recall_step=MOLT_STRATEGY_RECALL_STEP,
        decay_step=MOLT_STRATEGY_DECAY_STEP,
        abstraction_step=MOLT_STRATEGY_ABSTRACTION_STEP,
        recall_floor=MOLT_STRATEGY_RECALL_FLOOR,
        decay_ceiling=MOLT_STRATEGY_DECAY_CEILING,
        abstraction_floor=MOLT_STRATEGY_ABSTRACTION_FLOOR,
        low_importance_threshold=MOLT_STRATEGY_LOW_IMPORTANCE_THRESHOLD,
        initial_recall_threshold=MOLT_STRATEGY_INITIAL_RECALL_THRESHOLD,
        initial_decay_rate=MOLT_STRATEGY_INITIAL_DECAY_RATE,
        initial_abstraction_threshold=MOLT_STRATEGY_INITIAL_ABSTRACTION_THRESHOLD,
        history_limit=MOLT_STRATEGY_HISTORY_LIMIT,
    )


class MoltController(IMoltController):
    def __init__(
        self,
        mongo_repo=None,
        neo4j_repo=None,
        decay_scheduler=None,
        abstraction_engine=None,
        version_manager=None,
        feedback_service=None,
        sensory_cache: ISensoryCache | None = None,
        working_memory: IWorkingMemory | None = None,
        strategy_config: Optional[MoltStrategyConfig] = None,
        history_store: Optional[IStrategyHistoryStore] = None,
    ):
        self._mongo_repo = mongo_repo
        self._neo4j_repo = neo4j_repo
        self._decay_scheduler = decay_scheduler
        self._abstraction_engine = abstraction_engine
        self._version_manager = version_manager
        self._feedback_service = feedback_service
        self._sensory_cache = sensory_cache
        self._working_memory = working_memory
        self._config = strategy_config or default_strategy_config()
        self._history_store = history_store or InMemoryStrategyHistoryStore(
            limit=self._config.history_limit
        )
        self._recall_hit_rate = 0.5
        self._recall_threshold = self._config.initial_recall_threshold
        self._decay_rate = self._config.initial_decay_rate
        self._abstraction_trigger_threshold = self._config.initial_abstraction_threshold

    def start(self) -> None:
        if self._sensory_cache is not None:
            self._sensory_cache.start()

    def stop(self) -> None:
        if self._sensory_cache is not None:
            self._sensory_cache.stop()
        if self._working_memory is not None:
            self._working_memory.clear()

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
            "memory_tiers": self._memory_tier_stats(),
            "strategy": self.get_strategy(),
            "strategy_history": self._strategy_history_metrics(),
        }

    def evaluate_and_adjust(self) -> Dict[str, Any]:
        started_at = time.perf_counter()
        metrics = self.collect_metrics()
        adjustments = self._collect_adjustments(metrics)
        strategy = self.get_strategy()
        record = self._history_record(metrics, adjustments, strategy, started_at)
        if adjustments:
            self._history_store.append(record)
            logger.info(
                "molt strategy adjusted",
                extra={
                    "molt_adjustment_id": record["adjustment_id"],
                    "molt_adjustments": adjustments,
                    "molt_strategy": strategy,
                },
            )
        else:
            logger.debug(
                "molt strategy unchanged",
                extra={"molt_metrics": metrics, "molt_strategy": strategy},
            )
        return {
            "metrics": metrics,
            "adjustments": adjustments,
            "strategy": strategy,
            "history": self._history_store.recent(RECENT_HISTORY_SIZE),
            "timestamp": record["timestamp"],
        }

    def get_strategy(self) -> Dict[str, float]:
        return {
            "recall_threshold": self._recall_threshold,
            "decay_rate": self._decay_rate,
            "abstraction_trigger_threshold": self._abstraction_trigger_threshold,
        }

    def _collect_adjustments(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = (
            self._adjust_recall(metrics),
            self._adjust_decay(metrics),
            self._adjust_abstraction(metrics),
        )
        return [item for item in candidates if item is not None]

    def _adjust_recall(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if metrics["recall_hit_rate"] >= self._config.recall_hit_rate_threshold:
            return None
        previous = self._recall_threshold
        self._recall_threshold = max(
            self._config.recall_floor, previous - self._config.recall_step
        )
        return self._adjustment(
            "recall_threshold", previous, self._recall_threshold, "low_recall"
        )

    def _adjust_decay(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if metrics["avg_importance"] >= self._config.low_importance_threshold:
            return None
        previous = self._decay_rate
        self._decay_rate = min(self._config.decay_ceiling, previous + self._config.decay_step)
        return self._adjustment("decay_rate", previous, self._decay_rate, "low_importance")

    def _adjust_abstraction(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if metrics["abstraction_rate"] >= self._config.min_abstraction_rate:
            return None
        previous = self._abstraction_trigger_threshold
        self._abstraction_trigger_threshold = max(
            self._config.abstraction_floor, previous - self._config.abstraction_step
        )
        return self._adjustment(
            "abstraction_trigger_threshold",
            previous,
            self._abstraction_trigger_threshold,
            "low_abstraction",
        )

    def _adjustment(
        self, target: str, previous: float, value: float, reason: str
    ) -> Dict[str, Any]:
        return {"type": target, "from": previous, "to": value, "value": value, "reason": reason}

    def _history_record(
        self,
        metrics: Dict[str, Any],
        adjustments: List[Dict[str, Any]],
        strategy: Dict[str, float],
        started_at: float,
    ) -> Dict[str, Any]:
        return {
            "adjustment_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "metrics": metrics,
            "adjustments": adjustments,
            "strategy": strategy,
        }

    def _strategy_history_metrics(self) -> Dict[str, Any]:
        return {
            "total": self._history_store.count(),
            "recent": self._history_store.recent(RECENT_HISTORY_SIZE),
        }

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

    def _memory_tier_stats(self) -> Dict[str, Any]:
        return {
            "sensory": self._sensory_cache.stats() if self._sensory_cache else {},
            "working": self._working_memory.stats() if self._working_memory else {},
        }
