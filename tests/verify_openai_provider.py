import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from constants import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_ORGANIZATION,
    OPENAI_PROJECT,
    OPENAI_PROXY_URL,
    OPENAI_TIMEOUT,
)
from remote_providers import OpenAIModelProvider, OpenAISettings


def _none_if_empty(value: str):
    return value.strip() or None


def _parse_timeout(value: str):
    if value.strip() == "":
        return None
    return float(value)


def main() -> int:
    if OPENAI_API_KEY.strip() == "":
        print("OPENAI_API_KEY is empty. Please set it in .env")
        return 1

    settings = OpenAISettings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        organization=_none_if_empty(OPENAI_ORGANIZATION),
        project=_none_if_empty(OPENAI_PROJECT),
        proxy_url=_none_if_empty(OPENAI_PROXY_URL),
        timeout=_parse_timeout(OPENAI_TIMEOUT),
    )

    provider = OpenAIModelProvider(settings)

    try:
        models = provider.list_models()
    except Exception as exc:
        print(f"Failed to list models: {exc}")
        return 1

    if not models:
        print("No models returned from OpenAI.")
        return 1

    model = models[0]

    print(f"OpenAI base_url: {OPENAI_BASE_URL}")
    print(f"Using model: {model}")

    try:
        response = provider.chat("Hello, reply with OK.", model=model)
        print("Chat response:")
        print(response)
    except Exception as exc:
        print(f"Chat failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
