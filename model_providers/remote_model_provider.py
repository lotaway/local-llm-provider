from dataclasses import dataclass
from typing import List
from providers.provider_base import ModelProvider


@dataclass(frozen=True)
class RemoteProviderConfig:
    provider_id: str
    api_key: str
    model_names: List[str]


class RemoteModelProvider(ModelProvider):
    def __init__(self, config: RemoteProviderConfig):
        self._config = config

    def list_models(self) -> List[str]:
        if self.is_available():
            return list(self._config.model_names)
        return []

    def is_available(self) -> bool:
        return self._config.api_key.strip() != "" and len(self._config.model_names) > 0
