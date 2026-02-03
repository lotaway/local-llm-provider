from dataclasses import dataclass
from typing import Callable, Dict, List
from utils import discover_models
from .provider_base import ModelProvider
from .local_model_provider import LocalModelProvider
from .remote_model_provider import RemoteModelProvider, RemoteProviderConfig
from .openai_model_provider import OpenAIModelProvider, OpenAISettings


@dataclass(frozen=True)
class ProviderRegistrySettings:
    poe_api_key: str
    openai_api_key: str
    poe_default_model: str
    openai_base_url: str
    openai_organization: str
    openai_project: str
    openai_proxy_url: str
    openai_timeout: str


class ModelProviderRegistry:
    def __init__(self, providers: List[ModelProvider]):
        self._providers = providers

    def list_models(self) -> List[str]:
        models: List[str] = []
        for provider in self._providers:
            if provider.is_available():
                models.extend(provider.list_models())
        return models


def _model_list_from_default(value: str) -> List[str]:
    if value.strip() == "":
        return []
    return [value]


def _none_if_empty(value: str):
    return value.strip() or None


def _parse_timeout(value: str):
    if value.strip() == "":
        return None
    return float(value)


def parse_model_list(value: str, fallback: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if items:
        return items
    if fallback.strip() != "":
        return [fallback]
    return []


def build_default_registry(
    settings: ProviderRegistrySettings,
    model_discovery: Callable[[], Dict[str, str]] = discover_models,
) -> ModelProviderRegistry:
    providers: List[ModelProvider] = [
        LocalModelProvider(model_discovery=model_discovery),
        RemoteModelProvider(
            RemoteProviderConfig(
                provider_id="poe",
                api_key=settings.poe_api_key,
                model_names=_model_list_from_default(settings.poe_default_model),
            )
        ),
        OpenAIModelProvider(
            OpenAISettings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                organization=_none_if_empty(settings.openai_organization),
                project=_none_if_empty(settings.openai_project),
                proxy_url=_none_if_empty(settings.openai_proxy_url),
                timeout=_parse_timeout(settings.openai_timeout),
            )
        ),
    ]
    return ModelProviderRegistry(providers)
