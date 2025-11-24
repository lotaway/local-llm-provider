import openai
import os
import httpx
import json
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from fastapi import Request

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
            base_url="https://api.poe.com/v1",
            http_client=http_client,
        )

    def ping(self):
        return self.chat("Hello, introduce yourself.")

    def chat(self, message, model="Claude-Sonnet-4.5"):
        chat = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        return chat.choices[0].message.content
    

    async def handle_request(self, path, request: Request):
        """Process original URL, remove /poe path, forward to API with API_KEY"""
        body_data = await request.json()
        try:
            resp = self.client.post(path=path, body=body_data or {}, cast_to=httpx.Response, stream=body_data.get("stream", False))
            resp.raise_for_status()
            return resp
        except Exception as e:
            return f"data: {json.dumps({'error': str(e)})}\n\n"
