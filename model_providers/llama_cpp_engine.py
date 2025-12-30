import asyncio
import subprocess
import httpx
import json
import os
import signal
import socket
from typing import AsyncGenerator
from model_providers import InferenceEngine


class LlamaCppEngine(InferenceEngine):
    def __init__(
        self,
        model_path: str,
        project_root: str,
        n_ctx: int = 4096,
        batch_size: int = 2048,
        n_gpu_layers: int = -1,
    ):
        self.model_path = model_path
        self.project_root = project_root
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.batch_size = batch_size
        self.server_process = None
        self.port = self._find_free_port()
        self.base_url = f"http://localhost:{self.port}"
        self._start_server()

    @property
    def model_type(self) -> str:
        return "gguf"

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self):
        binary_path = os.path.join(
            self.project_root, "llama.cpp", "bin", "llama-server"
        )
        if not os.path.exists(binary_path):
            binary_path = os.path.join(
                self.project_root, "llama.cpp", "build", "bin", "llama-server"
            )

        cmd = [
            binary_path,
            "-m",
            self.model_path,
            "-c",
            str(self.n_ctx),
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "999",
            "--batch-size",
            str(self.batch_size),
        ]
        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid
        )
        print(
            f"Starting llama-server on port {self.port} with process {self.server_process.pid}"
        )

    async def _wait_for_server(self, timeout=60):
        async with httpx.AsyncClient() as client:
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1)
        return False

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        from constants import CHUNK_SIZE

        if not await self._wait_for_server():
            raise RuntimeError("llama-server failed to start")

        payload = {"prompt": prompt, "stream": True, "cache_prompt": True, **kwargs}

        buffer = []
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.base_url}/completion", json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        content = data.get("content", "")
                        if content:
                            buffer.append(content)

                        if len(buffer) >= CHUNK_SIZE or data.get("stop"):
                            if buffer:
                                yield "".join(buffer)
                                buffer = []

                        if data.get("stop"):
                            break

    def unload(self):
        if self.server_process:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=10)
            except Exception as e:
                print(f"Error killing llama-server: {e}")
                if self.server_process:
                    self.server_process.kill()
            self.server_process = None
