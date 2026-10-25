"""
Domain: Model selection domain service.
Encapsulates model selection, engine matching core business logic.
"""
from typing import List, Optional
from domain.model.entities import Model, ModelId, ModelPath, ModelType, EngineType
from domain.model.repositories import ModelRepository


class ModelSelectionService:
    """Model selection domain service."""

    def __init__(self, repository: ModelRepository):
        self._repository = repository

    def list_available(self) -> List[Model]:
        """List all available models."""
        return self._repository.list_models()

    def get_model(self, model_id: ModelId) -> Optional[Model]:
        """Get specific model."""
        return self._repository.get_model(str(model_id))

    def resolve_model(self, model_id: str, preferred_engine: EngineType = EngineType.AUTO) -> Model:
        """Resolve model (core business logic: architecture compatibility check, engine fallback)."""
        model = self._repository.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")

        if preferred_engine != EngineType.AUTO:
            if not model.supports_engine(preferred_engine):
                raise EngineIncompatibleError(
                    f"Model {model_id} does not support engine {preferred_engine.value}"
                )
            model.engine_type = preferred_engine
        else:
            model.engine_type = model.preferred_engine()

        return model

    def find_by_path(self, path: ModelPath) -> Optional[Model]:
        """Find model by path."""
        return self._repository.find_by_path(path.value)

    def refresh_catalog(self) -> None:
        """Refresh model catalog."""
        self._repository.refresh()


class TokenBudgetService:
    """Token budget domain service."""

    def __init__(self, default_context_length: int = 8192, reserve_tokens: int = 1024):
        self._default_context_length = default_context_length
        self._reserve_tokens = reserve_tokens

    def calculate_budget(self, context_length: int, max_new_tokens: int) -> int:
        """Calculate available token budget for history messages."""
        available = context_length - max_new_tokens - self._reserve_tokens
        return max(0, available)

    def truncate_messages(
        self, 
        messages: List[dict], 
        token_counter: callable,
        budget: int
    ) -> List[dict]:
        """Truncate messages to fit budget (preserve system message)."""
        if not messages:
            return []

        system_msg = None
        start_idx = 0
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            start_idx = 1

        history = messages[start_idx:]

        while history:
            test_msgs = ([system_msg] if system_msg else []) + history
            total_tokens = sum(token_counter(msg) for msg in test_msgs)

            if total_tokens <= budget:
                break

            if len(history) <= 1:
                break

            history.pop(0)

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(history)
        return result


class ModelNotFoundError(Exception):
    """Model not found error."""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class EngineIncompatibleError(Exception):
    """Engine incompatible error."""
    def __init__(self, message: str):
        super().__init__(message)