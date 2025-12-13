import contextlib
import json
import os
import shutil
import socket
import subprocess
import time
from typing import Optional
import urllib

import psutil
import requests

from llm_baseclient.config import OLLAMA_PORT, VLLM_BASE_URL, VLLM_PORT, vllm_default_command
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

    def _is_port_open(self, port: int) -> bool:
        """Checks if a localhost port is occupied."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("localhost", port)) == 0

    def _kill_inference_engines(self, targets: set[str]) -> None:
        """
        Scans system processes owned by the CURRENT USER for Ollama or vLLM signatures
        and force-kills them. Relies on API calls for system-level services.
        """
        current_pid = os.getpid()
        current_user = psutil.Process().username()

        # Unload Ollama models via API (secure, cross-platform)
        if "ollama" in targets:
            try:
                with urllib.request.urlopen(f"http://localhost:{OLLAMA_PORT}/api/ps", timeout=5) as response:
                    data = json.load(response)
                for model_info in data.get("models", []):
                    model_name = model_info["name"]
                    logger.warning(f"Unloading model via API: {model_name}")
                    unload_data = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
                    req = urllib.request.Request(
                        f"http://localhost:{OLLAMA_PORT}/api/generate", data=unload_data, headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(req, timeout=5)
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
                pass

        procs_to_kill = []
        # Include 'username' in fetch list to filter permissions
        for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
            try:
                if not proc.info["cmdline"]:
                    continue

                # PERMISSION CHECK: Only target processes we own
                if proc.info["username"] != current_user:
                    continue

                cmd_str = " ".join(proc.info["cmdline"]).lower()

                if any(t in cmd_str for t in targets):
                    if proc.info["pid"] == current_pid:
                        continue
                    procs_to_kill.append(proc)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if procs_to_kill:
            # Phase 1: Graceful Termination
            for proc in procs_to_kill:
                try:
                    logger.warning(f"Stopping CPU/GPU hoarder: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for processes to exit (increased timeout)
            _, alive = psutil.wait_procs(procs_to_kill, timeout=5)

            # Phase 2: Force Kill (SIGKILL) - no mercy for zombies & stubborn processes
            for proc in alive:
                with contextlib.suppress(psutil.NoSuchProcess):
                    proc.kill()

            # Wait for defragmentation of GPU memory
            psutil.wait_procs(alive, timeout=2)

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
        logger.info(f"Waiting for local inference server at {url}...")
        while time.time() - start_time < timeout:
            try:
                requests.get(url, timeout=1.5)
                logger.info(f"...inference server ready at {url}.")
                return True
            except requests.ConnectionError:
                time.sleep(2)
        return False

    def _spawn_server(self, cmd: list[str], health_check_url: str, install_hint: str) -> None:
        """Spawns a server process, waits for health, and captures stderr on failure."""
        if not shutil.which(cmd[0]):
            raise RuntimeError(f"Command not found: {cmd[0]}. {install_hint}")

        try:
            self._process = subprocess.Popen(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to spawn server: {e}")

        if not self._wait_for_server(health_check_url):
            _, stderr = self._process.communicate()
            raise RuntimeError(f"Server failed to start. Stderr: {stderr.decode().strip()}")

    def ensure_vllm(self, model_name: str, vllm_cmd: Optional[str] = None) -> None:
        """Ensures a vLLM server is running with the specified model."""
        if self._get_running_vllm_model(VLLM_BASE_URL) == model_name:
            return

        logger.info(f"Switching to vLLM ({model_name})...")
        self._kill_inference_engines(targets={"vllm", "ollama", "ollama runner"})

        # NOTE: --allow-remote-code is needed for some custom models
        vllm_url = f"http://localhost:{VLLM_PORT}/v1/models"
        if not vllm_cmd:
            vllm_cmd = vllm_default_command(model_name)
        self._spawn_server(cmd=vllm_cmd, health_check_url=vllm_url, install_hint="Install via: pip install vllm")

    def ensure_ollama(self, model_name: str) -> None:
        """Ensures an Ollama server is running."""
        subprocess.call(["ollama", "pull", model_name], stderr=subprocess.DEVNULL)
        if self._is_port_open(OLLAMA_PORT):
            # If Ollama is running, just ensure vLLM is off
            self._kill_inference_engines(targets={"vllm"})
            return

        logger.info("Switching to Ollama...")
        self._kill_inference_engines(targets={"vllm", "ollama", "ollama runner"})  # kill ollama zombies
        self._spawn_server(
            cmd=["ollama", "serve"], health_check_url=f"http://localhost:{OLLAMA_PORT}", install_hint="Install via: https://ollama.com"
        )
