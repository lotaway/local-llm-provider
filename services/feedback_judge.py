from dataclasses import dataclass
from typing import Dict, Any, List
import re

from interfaces.feedback_judge_types import IFeedbackJudge


@dataclass
class FeedbackJudgeConfig:
    min_confidence: float = 0.2
    max_confidence: float = 0.95
    neutral_confidence: float = 0.1


class FeedbackJudge(IFeedbackJudge):
    NEGATIVE_HINTS = [
        "不对",
        "错误",
        "不准确",
        "答非所问",
        "不满意",
        "不好",
        "糟糕",
        "你错了",
        "无效",
        "不行",
        "不合理",
        "不靠谱",
    ]
    POSITIVE_HINTS = [
        "谢谢",
        "很棒",
        "不错",
        "很好",
        "有帮助",
        "解决了",
        "可以",
        "满意",
        "正确",
        "靠谱",
    ]
    STYLE_HINTS = [
        "太长",
        "太短",
        "简洁",
        "详细",
        "口吻",
        "语气",
        "风格",
        "格式",
    ]
    TOOL_HINTS = [
        "工具",
        "搜索",
        "检索",
        "不用",
        "不要用",
        "应该用",
        "别用",
        "改用",
    ]
    REASONING_HINTS = [
        "理由",
        "推理",
        "逻辑",
        "过程",
        "步骤",
        "解释",
    ]

    def __init__(self, config: FeedbackJudgeConfig | None = None):
        self.config = config or FeedbackJudgeConfig()

    def evaluate(
        self, user_input: str, agent_response: str, context_window: List[str]
    ) -> Dict[str, Any]:
        text = (user_input or "").strip().lower()
        if not text:
            return self._neutral_signal()

        pos = self._contains_any(text, self.POSITIVE_HINTS)
        neg = self._contains_any(text, self.NEGATIVE_HINTS)
        style = self._contains_any(text, self.STYLE_HINTS)
        tool = self._contains_any(text, self.TOOL_HINTS)
        reasoning = self._contains_any(text, self.REASONING_HINTS)

        feedback_detected = pos or neg or style or tool or reasoning
        if not feedback_detected:
            return self._neutral_signal()

        if pos and neg:
            signal_type = "mixed"
        elif neg:
            signal_type = "negative"
        elif pos:
            signal_type = "positive"
        else:
            signal_type = "neutral"

        targets = []
        if style:
            targets.append("style")
        if tool:
            targets.append("tool_choice")
        if reasoning:
            targets.append("reasoning")
        if not targets and signal_type in {"positive", "negative", "mixed"}:
            targets.append("final_answer")

        confidence = self._estimate_confidence(pos, neg, style, tool, reasoning)
        action_hint = self._action_hint(signal_type, text)

        return {
            "feedback_detected": True,
            "type": signal_type,
            "confidence": confidence,
            "targets": targets,
            "action_hint": action_hint,
        }

    def _neutral_signal(self) -> Dict[str, Any]:
        return {
            "feedback_detected": False,
            "type": "neutral",
            "confidence": self.config.neutral_confidence,
            "targets": [],
            "action_hint": [],
        }

    def _contains_any(self, text: str, hints: List[str]) -> bool:
        return any(hint in text for hint in hints)

    def _estimate_confidence(
        self, pos: bool, neg: bool, style: bool, tool: bool, reasoning: bool
    ) -> float:
        score = 0.2
        for flag in (pos, neg, style, tool, reasoning):
            if flag:
                score += 0.15
        return max(self.config.min_confidence, min(self.config.max_confidence, score))

    def _action_hint(self, signal_type: str, text: str) -> List[str]:
        hints = []
        if signal_type == "negative":
            hints.append("penalty")
        elif signal_type == "positive":
            hints.append("prefer")
        elif signal_type == "mixed":
            hints.append("penalty")
            hints.append("prefer")

        if signal_type in {"negative", "mixed"} and re.search(r"错误|不对|无效|不准确", text):
            hints.append("invalidate")
        return list(dict.fromkeys(hints))
