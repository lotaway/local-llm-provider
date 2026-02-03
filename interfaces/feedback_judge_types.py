from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IFeedbackJudge(ABC):
    @abstractmethod
    def evaluate(
        self, user_input: str, agent_response: str, context_window: List[str]
    ) -> Dict[str, Any]:
        ...
