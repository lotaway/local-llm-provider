import asyncio
import subprocess
import httpx
import json
import os
import signal
import socket
import atexit
from typing import AsyncGenerator
from .inference_engine import InferenceEngine


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
        atexit.register(self.unload)

    @property
    def model_type(self) -> str:
        return "gguf"

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _cleanup_stale_processes(self):
        try:
            if os.getenv("LLAMA_CPP_USE_DOCKER", "false").lower() == "true":
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", "llp-llama-cpp-server"],
                        capture_output=True,
                    )
                except:
                    pass

            current_pid = self.server_process.pid if self.server_process else -1
            model_filename = os.path.basename(self.model_path)
            cmd = ["pgrep", "-f", f"llama-server.*{model_filename}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                pids = result.stdout.strip().split("\n")
                for pid_str in pids:
                    pid = int(pid_str)
                    if pid != current_pid and pid != os.getpid():
                        print(f"Cleaning up orphaned llama-server (PID: {pid})")
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except:
                            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            print(f"Warning during stale process cleanup: {e}")

    def _start_server(self):
        self._cleanup_stale_processes()

        mode = os.getenv("LLAMA_CPP_RUN_MODE", "docker").lower()
        image = os.getenv(
            "LLAMA_CPP_IMAGE",
            "rocm/llama.cpp:llama.cpp-b6356_rocm6.4.3_ubuntu24.04_full",
        )
        gfx_version = os.getenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        models_dir = os.getenv(
            "LLAMA_CPP_MODELS_DIR", os.path.join(self.project_root, "models")
        )
        if os.path.isabs(self.model_path):
            rel_model_path = os.path.relpath(self.model_path, models_dir)
        else:
            full_path = os.path.normpath(
                os.path.join(self.project_root, self.model_path)
            )
            rel_model_path = os.path.relpath(full_path, models_dir)

        if mode == "compose":
            # Start via Docker Compose
            # We use environment variables to "dynamically" configure the compose service
            compose_file = os.path.join(self.project_root, "docker-compose.env.yml")
            cmd = ["docker-compose", "-f", compose_file, "up", "-d", "llama.cpp"]

            run_env = os.environ.copy()
            run_env["LLAMA_MODEL_RELATIVE_PATH"] = rel_model_path
            run_env["LLAMA_N_GPU_LAYERS"] = str(
                self.n_gpu_layers if self.n_gpu_layers >= 0 else 999
            )
            run_env["LLAMA_N_CTX"] = str(self.n_ctx)
            run_env["LLAMA_BATCH_SIZE"] = str(self.batch_size)
            run_env["LLAMA_CPP_PORT"] = str(self.port)
            run_env["HSA_OVERRIDE_GFX_VERSION"] = gfx_version
            run_env["LLAMA_CPP_IMAGE"] = image

            print(f"Starting llama-server via Docker Compose (Model: {rel_model_path})")
            subprocess.run(cmd, env=run_env, check=True)

            # We monitor the container with docker logs
            # This is a dummy process to satisfy the engine's tracking
            self.server_process = subprocess.Popen(
                ["docker", "logs", "-f", "llp-llama-cpp"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        elif mode == "docker":
            # Start via standalone docker run
            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                "llp-llama-cpp-server",
                "--devices",
                "/dev/kfd",
                "--devices",
                "/dev/dri",
                "-v",
                f"{models_dir}:/models",
                "-p",
                f"{self.port}:8080",
                "-e",
                f"HSA_OVERRIDE_GFX_VERSION={gfx_version}",
                image,
                "llama-server",
                "-m",
                f"/models/{rel_model_path}",
                "-c",
                str(self.n_ctx),
                "--port",
                "8080",
                "--host",
                "0.0.0.0",
                "-ngl",
                str(self.n_gpu_layers) if self.n_gpu_layers >= 0 else "999",
                "--batch-size",
                str(self.batch_size),
                "-fa",
                "on",
            ]
            print(f"Starting llama-server via Docker: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
        else:
            # Start via local process
            binary_path = os.path.join(
                self.project_root, "llama.cpp", "bin", "llama-server"
            )
            if not os.path.exists(binary_path):
                binary_path = os.path.join(
                    self.project_root, "llama.cpp", "build", "bin", "llama-server"
                )

            env = os.environ.copy()
            env["HSA_OVERRIDE_GFX_VERSION"] = gfx_version

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
                "-fa",
                "on",
            ]

            run_env = os.environ.copy()
            run_env["HSA_OVERRIDE_GFX_VERSION"] = gfx_version

            log_file = os.path.join(self.project_root, "logs", "llama_server.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            f = open(log_file, "w")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=run_env,
            )
            f.close()
            print(f"Starting llama-server local process. Logs: {log_file}")

    async def _wait_for_server(self, timeout=120):
        async with httpx.AsyncClient(trust_env=False) as client:
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    url = self.base_url.replace("localhost", "127.0.0.1")
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        try:
                            data = json.loads(resp.text)
                            if data.get("status") == "ok":
                                return True
                        except json.JSONDecodeError:
                            print(
                                f"DEBUG: Health check returned 200 but invalid JSON: {resp.text}"
                            )
                    else:
                        print(
                            f"DEBUG: Status {resp.status_code} from health check: {resp.text}"
                        )
                except Exception:
                    pass
                await asyncio.sleep(1)
        return False

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        from constants import CHUNK_SIZE

        if not await self._wait_for_server():
            raise RuntimeError("llama-server failed to start")

        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])

        payload = {
            "messages": messages,
            "stream": True,
            "model": "gpt-3.5-turbo",  # placeholder
            **{k: v for k, v in kwargs.items() if k not in ["messages"]},
        }

        buffer = []
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            url = self.base_url.replace("localhost", "127.0.0.1")
            async with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload
            ) as response:
                if response.status_code != 200:
                    error_data = await response.aread()
                    print(
                        f"DEBUG: llama-server returned {response.status_code}: {error_data.decode()}"
                    )
                    raise RuntimeError(f"llama-server error: {error_data.decode()}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        line_content = line[6:].strip()
                        if line_content == "[DONE]":
                            break

                        try:
                            data = json.loads(line_content)
                            choices = data.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            reasoning = delta.get("reasoning_content")
                            finish_reason = choices[0].get("finish_reason")

                            if reasoning:
                                if buffer:
                                    yield "".join(buffer)
                                    buffer = []
                                yield {"reasoning_content": reasoning}
                            elif content:
                                buffer.append(content)

                            if (finish_reason and finish_reason != "null") or len(
                                buffer
                            ) >= CHUNK_SIZE:
                                yield {
                                    "content": "".join(buffer),
                                    "finish_reason": finish_reason,
                                }
                                buffer = []
                        except json.JSONDecodeError:
                            continue

    def unload(self):
        if self.server_process:
            try:
                print(f"Stopping llama-server...")
                if os.getenv("LLAMA_CPP_USE_DOCKER", "false").lower() == "true":
                    subprocess.run(
                        ["docker", "stop", "llp-llama-cpp-server"], capture_output=True
                    )
                if self.server_process.poll() is None:
                    try:
                        pgid = os.getpgid(self.server_process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        self.server_process.wait(timeout=5)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(
                                os.getpgid(self.server_process.pid), signal.SIGKILL
                            )
                        except:
                            pass
            except Exception as e:
                print(f"Error killing llama-server: {e}")
            finally:
                self.server_process = None
