from dataclasses import dataclass
from typing import Optional, Callable
import httpx
import openai
from fastapi import Request
from .provider_base import ModelProvider


class OpenAIProviderConfigError(ValueError):
    pass


class OpenAIProviderRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    base_url: str
    organization: Optional[str] = None
    project: Optional[str] = None
    proxy_url: Optional[str] = None
    timeout: Optional[float] = None


class OpenAIModelProvider(ModelProvider):
    def __init__(
        self,
        settings: OpenAISettings,
        http_client_factory: Optional[Callable[[OpenAISettings], httpx.Client]] = None,
    ):
        self._settings = self._validate_settings(settings)
        self._client_factory = http_client_factory or self._build_http_client
        self._client = None

    def ping(self, model: str):
        return self.chat("Hello, introduce yourself.", model=model)

    def chat(self, message: str, model: str):
        chat = self._get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        return chat.choices[0].message.content

    def list_models(self):
        if not self.is_available():
            return []
        data = self._get_client().models.list()
        return self._extract_model_ids(data)

    def is_available(self) -> bool:
        return self._settings.api_key.strip() != ""

    async def handle_request(
        self, path: str, request: Request, body_data: dict | None = None
    ):
        if body_data is None:
            body_data = await request.json()
        request_body = self._normalize_body_data(body_data)
        try:
            resp = self._get_client().post(
                path=path,
                body=request_body,
                cast_to=httpx.Response,
                stream=request_body.get("stream", False),
            )
            resp.raise_for_status()
            return resp
        except Exception as exc:
            raise OpenAIProviderRequestError(str(exc)) from exc

    def _build_openai_client(
        self, http_client_factory: Callable[[OpenAISettings], httpx.Client]
    ):
        http_client = http_client_factory(self._settings)
        return openai.OpenAI(
            api_key=self._settings.api_key,
            base_url=self._settings.base_url,
            organization=self._settings.organization,
            project=self._settings.project,
            http_client=http_client,
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.is_available():
            raise OpenAIProviderConfigError("api_key is required")
        self._client = self._build_openai_client(self._client_factory)
        return self._client

    def _build_http_client(self, settings: OpenAISettings) -> httpx.Client:
        transport = httpx.HTTPTransport(proxy=settings.proxy_url)
        return httpx.Client(transport=transport, timeout=settings.timeout)

    def _validate_settings(self, settings: OpenAISettings) -> OpenAISettings:
        if settings.base_url is None or settings.base_url.strip() == "":
            raise OpenAIProviderConfigError("base_url is required")
        if settings.timeout is not None and settings.timeout <= 0:
            raise OpenAIProviderConfigError("timeout must be positive")
        return settings

    def _extract_model_ids(self, data) -> list[str]:
        models = getattr(data, "data", [])
        return [m.id for m in models if getattr(m, "id", "").strip() != ""]

    def _normalize_body_data(self, body_data: dict | None) -> dict:
        if body_data is None:
            return {}
        if isinstance(body_data, dict):
            return body_data
        raise OpenAIProviderRequestError("request body must be a JSON object")
