from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from constants import LEARNING
from interfaces.evolution_types import IEvolutionDispatcher
from interfaces.memory_types import ISensoryCache, IWorkingMemory


@dataclass
class EvolutionConfig:
    confidence_threshold: float = 0.55
    min_signals_m2: int = 2
    min_signals_m3: int = 3
    m2_delta: float = 0.05
    m3_delta: float = 0.05
    history_limit: int = 200


class EvolutionDispatcher(IEvolutionDispatcher):
    def __init__(
        self,
        memory_repo=None,
        config: Optional[EvolutionConfig] = None,
        skill_researcher=None,
        skill_creator=None,
        sensory_cache: Optional[ISensoryCache] = None,
        working_memory: Optional[IWorkingMemory] = None,
    ):
        self._memory_repo = memory_repo
        self._config = config or EvolutionConfig()
        self._skill_researcher = skill_researcher
        self._skill_creator = skill_creator
        self._sensory_cache = sensory_cache
        self._working_memory = working_memory
        self._signal_pool: List[Dict[str, Any]] = []
        self._history: List[Dict[str, Any]] = []

    def enqueue(self, signal: Dict[str, Any]) -> None:
        if not signal or not signal.get("feedback_detected"):
            return
        self._signal_pool.append(signal)
        self._history.append(signal)
        if len(self._history) > self._config.history_limit:
            self._history = self._history[-self._config.history_limit :]
        self._admit_to_sensory_cache(signal)
        self._admit_to_working_memory(signal)

    def aggregate_and_apply(
        self,
        context: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        meta = meta or {}
        aggregated = self._aggregate_signals()
        adjustments = self._apply_adjustments(aggregated, context, meta)
        self._signal_pool = []
        return {
            "aggregated": aggregated,
            "adjustments": adjustments,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def _aggregate_signals(self) -> Dict[str, Any]:
        usable = [
            s
            for s in self._signal_pool
            if s.get("confidence", 0) >= self._config.confidence_threshold
        ]
        totals = self._count_target_totals(usable)
        return {
            "total_signals": len(self._signal_pool),
            "usable_signals": len(usable),
            "summary": {
                target: {"dominant": self._dominant_type(counts), "counts": counts}
                for target, counts in totals.items()
            },
        }

    def _count_target_totals(self, usable: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        totals: Dict[str, Dict[str, int]] = {}
        for signal in usable:
            for target in signal.get("targets", []):
                counts = totals.setdefault(target, {"positive": 0, "negative": 0, "mixed": 0})
                counts[signal.get("type", "neutral")] = counts.get(signal.get("type", "neutral"), 0) + 1
        return totals

    def _dominant_type(self, counts: Dict[str, int]) -> str:
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if not sorted_counts:
            return "neutral"
        if len(sorted_counts) > 1 and sorted_counts[0][1] == sorted_counts[1][1]:
            return "mixed"
        return sorted_counts[0][0]

    def _apply_adjustments(
        self, aggregated: Dict[str, Any], context: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        adjustments = {"m1": [], "m2": [], "m3": []}
        for target, info in aggregated.get("summary", {}).items():
            self._dispatch_dominant_adjustments(target, info, context, meta, adjustments)
        self._record_adjustment_history(adjustments, aggregated, context)
        return adjustments

    def _dispatch_dominant_adjustments(
        self, target: str, info: Dict[str, Any], context: Dict[str, Any], meta: Dict[str, Any], adjustments: Dict[str, Any]
    ) -> None:
        dominant = info.get("dominant", "neutral")
        if dominant == "neutral":
            return
        self._apply_m1(context, target, dominant, adjustments)
        if dominant == "mixed":
            return
        action = "penalty" if dominant == "negative" else "prefer"
        if self._count_consistent_signals(target, dominant) >= self._config.min_signals_m2:
            self._apply_m2(context, meta, target, action, adjustments)
        if self._count_consistent_signals(target, dominant) >= self._config.min_signals_m3:
            self._apply_m3(context, target, action, adjustments)

    def _record_adjustment_history(
        self, adjustments: Dict[str, Any], aggregated: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        if not (adjustments["m1"] or adjustments["m2"] or adjustments["m3"]):
            return
        history = context.setdefault("evolution_history", [])
        history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "aggregated": aggregated,
                "adjustments": adjustments,
            }
        )

    def _apply_m1(
        self, context: Dict[str, Any], target: str, dominant: str, adjustments: Dict[str, Any]
    ) -> None:
        entry = {
            "target": target,
            "type": dominant,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        context.setdefault("m1_feedback", []).append(entry)
        adjustments["m1"].append(entry)

    def _apply_m2(
        self,
        context: Dict[str, Any],
        meta: Dict[str, Any],
        target: str,
        action: str,
        adjustments: Dict[str, Any],
    ) -> None:
        delta = self._config.m2_delta if action == "prefer" else -self._config.m2_delta
        m2 = context.setdefault("m2_adjustments", {})
        m2[target] = m2.get(target, 0.0) + delta
        adjustments["m2"].append({"target": target, "delta": delta})

        if target == "tool_choice":
            self._adjust_agent_preferences(context, meta, delta, adjustments)

    def _adjust_agent_preferences(
        self,
        context: Dict[str, Any],
        meta: Dict[str, Any],
        delta: float,
        adjustments: Dict[str, Any],
    ) -> None:
        agent_type = context.get("last_agent_type")
        if not agent_type:
            return
        self._adjust_agent_type_preference(context, agent_type, delta, adjustments)
        self._adjust_skill_preferences(context, meta.get("skills_used", []), delta, adjustments)

    def _adjust_agent_type_preference(self, context, agent_type, delta, adjustments) -> None:
        prefs = context.setdefault("agent_type_preferences", {})
        prefs[agent_type] = self._clamp(prefs.get(agent_type, 0.5) + delta)
        adjustments["m2"].append(
            {"target": f"agent_type:{agent_type}", "delta": delta}
        )

    def _adjust_skill_preferences(self, context, skills_used, delta, adjustments) -> None:
        if not skills_used:
            return
        skill_prefs = context.setdefault("skill_preferences", {})
        for skill in skills_used:
            skill_prefs[skill] = self._clamp(skill_prefs.get(skill, 0.5) + delta)
            adjustments["m2"].append({"target": f"skill:{skill}", "delta": delta})
    def _apply_m3(
        self,
        context: Dict[str, Any],
        target: str,
        action: str,
        adjustments: Dict[str, Any],
    ) -> None:
        if not LEARNING or not self._memory_repo or target not in {"final_answer", "reasoning"}:
            return
        chunk_ids = context.get("last_retrieved_chunk_ids") or []
        if not chunk_ids:
            return
        delta = self._config.m3_delta if action == "prefer" else -self._config.m3_delta
        applied = self._increment_chunk_importance(chunk_ids, delta)
        if applied:
            adjustments["m3"].append({"delta": delta, "chunk_ids": applied})

    def _increment_chunk_importance(self, chunk_ids: List[str], delta: float) -> List[str]:
        applied = []
        for chunk_id in chunk_ids:
            try:
                self._memory_repo.increment_importance(chunk_id, delta)
                applied.append(chunk_id)
            except Exception:
                continue
        return applied

    def _count_consistent_signals(self, target: str, signal_type: str) -> int:
        count = 0
        for signal in self._history:
            if signal.get("confidence", 0) < self._config.confidence_threshold:
                continue
            if signal.get("type") != signal_type:
                continue
            if target in signal.get("targets", []):
                count += 1
        return count

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _admit_to_sensory_cache(self, signal: Dict[str, Any]) -> None:
        if self._sensory_cache is None:
            return
        self._sensory_cache.remember(f"signal:{uuid4().hex}", signal)

    def _admit_to_working_memory(self, signal: Dict[str, Any]) -> None:
        if self._working_memory is None:
            return
        payload = {
            "type": signal.get("type", "neutral"),
            "confidence": signal.get("confidence", 0.0),
        }
        for target in signal.get("targets", []):
            self._working_memory.focus(target, payload)

    async def trigger_skill_evolution(self, task_description: str, error_context: str) -> Dict[str, Any]:
        """Manually or automatically trigger the skill evolution loop."""
        if not self._skill_researcher or not self._skill_creator:
            return {"error": "Evolution services not configured"}

        research_result = await self._skill_researcher.research_missing_skill(task_description, error_context)
        if "error" in research_result:
            return research_result

        success = await self._skill_creator.create_and_deploy(research_result)

        return {
            "success": success,
            "skill_ir": research_result,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
