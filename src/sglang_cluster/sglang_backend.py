from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp

from src.sglang_cluster.config import SGLangConfig


class SGLangBackend:
    def __init__(
        self,
        node_id: int,
        port: int,
        config: SGLangConfig,
        project_root: Path,
        gpu_id: int | None,
    ):
        self.node_id = node_id
        self.port = port
        self.config = config
        self.project_root = project_root
        self.gpu_id = gpu_id
        self.base_url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._log_handle = None

    async def start(self):
        if self._process and self._process.poll() is None:
            return

        log_dir = self.project_root / self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"node_{self.node_id}_sglang_{self.port}.log"
        self._log_handle = log_path.open("w")

        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        local_python_path = self.project_root / "sglang" / "python"
        if self.config.use_local_checkout and local_python_path.exists():
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{local_python_path}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(local_python_path)
            )

        cmd = [
            self.config.python_executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model_path,
            "--port",
            str(self.port),
            "--tp-size",
            str(self.config.tp_size),
            "--mem-fraction-static",
            str(self.config.mem_fraction_static),
            "--context-length",
            str(self.config.context_length),
        ]
        if self.config.enable_metrics:
            cmd.append("--enable-metrics")

        self._process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=self._log_handle,
            stderr=self._log_handle,
            start_new_session=True,
        )
        await self._wait_until_ready()

    async def _wait_until_ready(self):
        deadline = time.monotonic() + self.config.startup_timeout_s
        headers = {"Authorization": "Bearer None"}
        last_error: str | None = None

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server for node {self.node_id} exited early. "
                    f"Check logs in {self.config.log_dir}."
                )

            try:
                timeout = aiohttp.ClientTimeout(total=3)
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(f"{self.base_url}/v1/models") as response:
                        if response.status < 500:
                            await asyncio.sleep(5)
                            return
            except Exception as exc:  # pragma: no cover - best effort readiness loop
                last_error = str(exc)

            await asyncio.sleep(1)

        raise TimeoutError(
            f"Timed out waiting for SGLang server on port {self.port}. "
            f"Last error: {last_error or 'unknown'}"
        )

    async def generate(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload = {
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
        headers = {"Authorization": "Bearer None"}
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if isinstance(data, list):
                    if not data:
                        raise RuntimeError("SGLang /generate returned an empty list.")
                    return data[0]
                return data

    async def get_radix_tree(self) -> Any:
        headers = {"Authorization": "Bearer None"}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(f"{self.base_url}/debug/get_radix_tree") as response:
                response.raise_for_status()
                return await response.json()

    async def stop(self):
        if self._process and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                self._process.terminate()

            try:
                await asyncio.to_thread(self._process.wait, 10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    self._process.kill()
                await asyncio.to_thread(self._process.wait)

        self._process = None

        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
