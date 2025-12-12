import atexit
import shutil
import socket
import subprocess
import time
from typing import Optional

import psutil
import requests

from llm_baseclient.config import OLLAMA_PORT, VLLM_BASE_URL, VLLM_GPU_UTIL, VLLM_PORT
from llm_baseclient.logger import get_logger

logger = get_logger()

# ----------------------------------- Server Manager --------------------------------- #

class _LocalServerManager:
    """
    Manages server processes for vLLM or Ollama.
    Assumes requested models to be available.
    Ensures server is terminated upon exit.
    Designed for single-GPU multi-backend usage.
    Allows dynamic switching between backends/models.
    """
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        atexit.register(self.stop_server)

    def stop_server(self) -> None:
        """Terminates any running local inference server."""
        if self._process:
            try:
                logger("Stopping local inference server...")
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                self._process.kill()
            self._process = None

    def _is_port_open(self, port: int) -> bool:
        """Checks if a localhost port is occupied."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) == 0

    def _kill_processes_on_ports(self, *ports: int) -> None:
        """Terminates any processes listening on the specified ports."""
        for port in ports:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        logger(f"Freeing Port {port} (PID {proc.pid})...")
                        proc.terminate()
                        proc.wait(timeout=5)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        pass

    def _get_running_vllm_model(self, base_url: str) -> Optional[str]:
        """Queries the vLLM /v1/models endpoint to check which model is loaded."""
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=1)
            if resp.status_code == 200 and (data := resp.json()).get("data"):
                return data["data"][0]["id"]
        except requests.RequestException:
            pass
        return None

    def _wait_for_server(self, url: str, timeout: int = 60) -> bool:
        """Health Check: Blocks until server returns 200 OK."""
        start_time = time.time()
        logger(f"Waiting for local inference server at {url}...")
        while time.time() - start_time < timeout:
            try:
                requests.get(url, timeout=1.5)
                logger(f"...inference server ready at {url}.")
                return True
            except requests.ConnectionError:
                time.sleep(2)
        return False

    def _spawn_server(self, cmd: list[str], health_check_url: str, install_hint: str) -> None:
        """Spawns a server process, waits for health, and captures stderr on failure."""
        if not shutil.which(cmd[0]):
            raise RuntimeError(f"Command not found: {cmd[0]}. {install_hint}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE # Capture stderr for debugging
            )
        except Exception as e:
            raise RuntimeError(f"Failed to spawn server: {e}")

        if not self._wait_for_server(health_check_url):
            _, stderr = self._process.communicate()
            self.stop_server()
            raise RuntimeError(f"Server failed to start. Logs:\n{stderr.decode('utf-8') if stderr else 'Unknown error'}")

    def ensure_vllm(self, model_name: str) -> None:
        """Ensures a vLLM server is running with the specified model."""
        if self._get_running_vllm_model(VLLM_BASE_URL) == model_name:
            return

        logger(f"Switching to vLLM ({model_name})...")
        self._kill_processes_on_ports(VLLM_PORT, OLLAMA_PORT)

        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name, "--port", str(VLLM_PORT),
            "--trust-remote-code", "--gpu-memory-utilization", str(VLLM_GPU_UTIL)
        ]
        self._spawn_server(
            cmd=vllm_cmd,
            health_check_url=f"http://localhost:{VLLM_PORT}/v1/models",
            install_hint="Install via: pip install vllm")

    def ensure_ollama(self) -> None:
        """Ensures an Ollama server is running."""
        if self._is_port_open(OLLAMA_PORT):
            # If Ollama is running, just ensure vLLM is off
            self._kill_processes_on_ports(VLLM_PORT)
            return

        logger("Switching to Ollama...")
        self._kill_processes_on_ports(OLLAMA_PORT, VLLM_PORT)
        self._spawn_server(
            cmd=["ollama", "serve"],
            health_check_url=f"http://localhost:{OLLAMA_PORT}",
            install_hint="Install via: https://ollama.com")
