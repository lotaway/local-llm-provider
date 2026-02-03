from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

from constants import LEARNING
from interfaces.evolution_types import IEvolutionDispatcher


@dataclass
class EvolutionConfig:
    confidence_threshold: float = 0.55
    min_signals_m2: int = 2
    min_signals_m3: int = 3
    m2_delta: float = 0.05
    m3_delta: float = 0.05
    history_limit: int = 200


class EvolutionDispatcher(IEvolutionDispatcher):
    def __init__(self, memory_repo=None, config: EvolutionConfig | None = None):
        self._memory_repo = memory_repo
        self._config = config or EvolutionConfig()
        self._signal_pool: List[Dict[str, Any]] = []
        self._history: List[Dict[str, Any]] = []

    def enqueue(self, signal: Dict[str, Any]) -> None:
        if not signal or not signal.get("feedback_detected"):
            return
        self._signal_pool.append(signal)
        self._history.append(signal)
        if len(self._history) > self._config.history_limit:
            self._history = self._history[-self._config.history_limit :]

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
        totals = {}
        usable = [
            s
            for s in self._signal_pool
            if s.get("confidence", 0) >= self._config.confidence_threshold
        ]
        for signal in usable:
            s_type = signal.get("type", "neutral")
            for target in signal.get("targets", []):
                if target not in totals:
                    totals[target] = {"positive": 0, "negative": 0, "mixed": 0}
                if s_type not in totals[target]:
                    totals[target][s_type] = 0
                totals[target][s_type] += 1

        summary = {}
        for target, counts in totals.items():
            dominant = self._dominant_type(counts)
            summary[target] = {
                "dominant": dominant,
                "counts": counts,
            }

        return {
            "total_signals": len(self._signal_pool),
            "usable_signals": len(usable),
            "summary": summary,
        }

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
        summary = aggregated.get("summary", {})
        for target, info in summary.items():
            dominant = info.get("dominant", "neutral")
            if dominant == "neutral":
                continue
            self._apply_m1(context, target, dominant, adjustments)

            if dominant == "mixed":
                continue

            action = "penalty" if dominant == "negative" else "prefer"
            if (
                self._count_consistent_signals(target, dominant)
                >= self._config.min_signals_m2
            ):
                self._apply_m2(context, meta, target, action, adjustments)

            if (
                self._count_consistent_signals(target, dominant)
                >= self._config.min_signals_m3
            ):
                self._apply_m3(context, target, action, adjustments)

        if adjustments["m1"] or adjustments["m2"] or adjustments["m3"]:
            history = context.setdefault("evolution_history", [])
            history.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "aggregated": aggregated,
                    "adjustments": adjustments,
                }
            )
        return adjustments

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
        prefs = context.setdefault("agent_type_preferences", {})
        prefs[agent_type] = self._clamp(prefs.get(agent_type, 0.5) + delta)
        adjustments["m2"].append(
            {"target": f"agent_type:{agent_type}", "delta": delta}
        )

        skills_used = meta.get("skills_used", [])
        if skills_used:
            skill_prefs = context.setdefault("skill_preferences", {})
            for skill in skills_used:
                skill_prefs[skill] = self._clamp(skill_prefs.get(skill, 0.5) + delta)
                adjustments["m2"].append(
                    {"target": f"skill:{skill}", "delta": delta}
                )

    def _apply_m3(
        self,
        context: Dict[str, Any],
        target: str,
        action: str,
        adjustments: Dict[str, Any],
    ) -> None:
        if not LEARNING or not self._memory_repo:
            return
        if target not in {"final_answer", "reasoning"}:
            return

        chunk_ids = context.get("last_retrieved_chunk_ids") or []
        if not chunk_ids:
            return

        delta = self._config.m3_delta if action == "prefer" else -self._config.m3_delta
        applied = []
        for chunk_id in chunk_ids:
            try:
                self._memory_repo.increment_importance(chunk_id, delta)
                applied.append(chunk_id)
            except Exception:
                continue
        if applied:
            adjustments["m3"].append({"delta": delta, "chunk_ids": applied})

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
