import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

class ComfyUIProvider:
    async def proxy_request(self, request: Request, target_host: str = "http://localhost:8188"):
        async with httpx.AsyncClient() as client:
            body_data = await request.body()
            headers = dict(request.headers)
            # response = await client.post(target_host, headers=headers, content=body_data)
            new_request = httpx.Request(request.method, target_host, headers=headers, content=body_data)
            response = await client.send(new_request)
            return StreamingResponse(response.aiter_bytes(), status_code=response.status_code, media_type=response.headers.get('content-type'))