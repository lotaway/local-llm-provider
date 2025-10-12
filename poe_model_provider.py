import openai
import os
import httpx
import json
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
    

    def handle_request(self, path, body_data=None):
        """Process original URL, remove /poe path, forward to API with API_KEY"""
        api_url = f"https://api.poe.com/{path}"
        print(f"Forwarding request to: {api_url}")
        headers = { 
            "Authorization": f"Bearer {self.client.api_key}",
            "Accept": "text/event-stream" 
        }
        
        def generate():
            try:
                with httpx.Client(timeout=None) as client:
                    with client.stream("POST", api_url, headers=headers, json=body_data or {}) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_text():
                            if chunk:
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return generate()
