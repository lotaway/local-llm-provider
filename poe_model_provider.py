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

    def handle_request(self, original_url, body_data=None):
        """Process original URL, remove /poe path, forward to API with API_KEY"""
        try:
            if isinstance(original_url, bytes):
                original_url = original_url.decode('utf-8')
            parsed = urlparse(original_url)
            path = parsed.path
            if path.startswith("/poe"):
                path = path[4:] or "/"
            api_url = f"https://api.poe.com{path}"
            if parsed.query:
                api_url += f"?{parsed.query}"
            print(f"Forwarding request to: {api_url}")
            response = self.client._client.post(
                url=api_url,
                headers={
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json=body_data or {}
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            print(f"API request failed with status {e.response.status_code}: {e}")
            raise
        except Exception as e:
            print(f"Error processing request: {e}")
            raise
