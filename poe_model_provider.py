import openai
import os
import httpx
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


class PoeModelProvider:
    def __init__(self):
        proxy_url = os.getenv("HTTP_PROXY")
        if proxy_url is None:
            raise ValueError("HTTP_PROXY environment variable is not set")
        api_key = os.getenv("POE_API_KEY")
        if api_key is None:
            raise ValueError("POE_API_KEY environment variable is not set")
        transport = httpx.HTTPTransport(proxy=proxy_url)
        http_client = httpx.Client(transport=transport)

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.poe.com",
            http_client=http_client,
        )

    def ping(self):
        print(self.chat("Hello, introduce yourself."))

    def chat(self, message, model="Claude-Sonnet-4.5"):
        chat = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        return chat.choices[0].message.content
    

    async def handle_request(self, path, body_data=None):
        """Process original URL, remove /poe path, forward to API with API_KEY"""
        try:
            api_url = f"https://api.poe.com{path}"
            print(f"Forwarding request to: {api_url}")
            headers = { "Authorization": f"Bearer {self.client.api_key}", "Accept": "text/event-stream" }
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", api_url, headers=headers, json=body_data or {}) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_text():
                        if chunk:
                            yield chunk
        except httpx.HTTPStatusError as e:
            print(f"API request failed with status {e.response.status_code}: {e}")
            raise
        except Exception as e:
            print(f"Error processing request: {e}")
            raise
