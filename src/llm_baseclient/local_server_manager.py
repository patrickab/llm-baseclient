import os
import shutil
import socket
import subprocess
import time
from typing import Optional

import psutil
import requests

from llm_baseclient.config import MAX_TOKEN_DEFAULT_VLLM, OLLAMA_PORT, VLLM_BASE_URL, VLLM_GPU_UTIL, VLLM_PORT
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
        Scans all system processes for Ollama or vLLM signatures and
        force-kills them to free GPU VRAM.
        """
        current_pid = os.getpid()

        # Unload Ollama models
        if "ollama" in targets:
            loaded_model_stdout = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True).stdout
            lines = loaded_model_stdout.strip().split("\n")[1:]  # remove whitespaces, split by line, remove first line
            loaded_models = [line.split()[0] for line in lines]
            for model in loaded_models:
                logger.warning(f"Force unloading {model}...")
                subprocess.run(
                    f'curl -s --show-error http://localhost:{OLLAMA_PORT}/api/generate -d \'{{"model": "{model}", "keep_alive": 0}}\'',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                )

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                # Skip processes without a command line (kernel threads, etc.)
                if not proc.info["cmdline"]:
                    continue

                # Convert command line list to a single string for easy searching
                cmd_str = " ".join(proc.info["cmdline"]).lower()

                # check if this process is one of our targets
                if any(t in cmd_str for t in targets):
                    # SAFETY CHECK: Don't kill yourself!
                    if proc.info["pid"] == current_pid:
                        continue

                    logger.warning(f"Force killing CPU/GPU hoarder: {proc.info['name']} (PID: {proc.info['pid']})")

                    # Send SIGKILL - no mercy for zombies
                    proc.kill()

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
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
            self._process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,  # Capture stderr for debugging
            )
        except Exception as e:
            raise RuntimeError(f"Failed to spawn server: {e}")

        if not self._wait_for_server(health_check_url):
            _, stderr = self._process.communicate()
            raise RuntimeError(f"Server failed to start. Logs:\n{stderr.decode('utf-8') if stderr else 'Unknown error'}")

    def ensure_vllm(self, model_name: str, max_tokens: Optional[int] = MAX_TOKEN_DEFAULT_VLLM) -> None:
        """Ensures a vLLM server is running with the specified model."""
        if self._get_running_vllm_model(VLLM_BASE_URL) == model_name:
            return

        logger.info(f"Switching to vLLM ({model_name})...")
        self._kill_inference_engines(targets={"vllm", "ollama", "ollama runner"})

        # NOTE: --allow-remote-code is needed for some custom models
        vllm_url = f"http://localhost:{VLLM_PORT}/v1/models"
        try:  # try GPU first
            vllm_cmd = [
                "vllm",
                "serve",
                model_name,
                "--port",
                str(VLLM_PORT),
                "--gpu-memory-utilization",
                str(VLLM_GPU_UTIL),
                "--max-model-len",  # max context length
                str(max_tokens),
            ]
            self._spawn_server(cmd=vllm_cmd, health_check_url=vllm_url, install_hint="Install via: pip install vllm")
        except Exception:
            try:  # fall back to CPU
                vllm_cmd = [
                    "vllm",
                    "serve",
                    model_name,
                    "--port",
                    str(VLLM_PORT),
                    "--device",
                    "cpu",
                    "--max-tokens",
                    str(max_tokens),
                ]
                self._spawn_server(cmd=vllm_cmd, health_check_url=vllm_url, install_hint="Install via: pip install vllm")
            except Exception:
                raise

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
