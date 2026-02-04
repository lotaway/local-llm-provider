from typing import Callable, Dict, List
from utils import discover_models
from providers.provider_base import ModelProvider


class LocalModelProvider(ModelProvider):
    def __init__(self, model_discovery: Callable[[], Dict[str, str]] = discover_models):
        self._model_discovery = model_discovery

    def list_models(self) -> List[str]:
        return list(self._model_discovery().keys())

    def is_available(self) -> bool:
        return True
