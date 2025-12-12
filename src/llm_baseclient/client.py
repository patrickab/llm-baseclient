"""
Swiss Army Knife LLM client.

Supports local inference on CPU/GPU via vLLM or Ollama.
Supports any LiteLLM-compatabile model - refer to https://models.litellm.ai/
Supports image batch processing with inputs as file paths, raw bytes, or base64 strings.
Supports dynamic switching between providers during runtime.

Automatically spawns & manages servers for local inference backends (vLLM, Ollama).
"""

import atexit
import base64
from pathlib import Path
import socket
import subprocess
import time
from typing import Dict, Iterator, List, Optional, Tuple, Union

from litellm import completion, embedding
from litellm.utils import EmbeddingResponse
from openai.types.chat import ChatCompletion
import psutil
import requests

from llm_baseclient.config import OLLAMA_PORT, VLLM_BASE_URL, VLLM_GPU_UTIL, VLLM_PORT

# ----------------------------------- Server Manager --------------------------------- #

class LocalServerManager:
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
            print("Stopping local inference server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()  # Force kill if graceful termination times out
            self._process = None

    def _is_port_open(self, port: int) -> bool:
        """Checks if a localhost port is occupied."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) == 0

    def _kill_processes_on_ports(self, *ports: int) -> None:
        """Terminates any processes listening on the specified ports."""
        for port in ports:
            for proc in psutil.process_iter(['pid', 'name']):
                # Iterate through all running processes to find and terminate those listening on the specified port.
                try:
                    for conn in proc.net_connections(kind='inet'):
                        if conn.laddr.port == port:
                            print(f"Port {port} occupied by PID {proc.info['pid']} ({proc.info['name']}). Terminating...")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()  # Force kill if graceful termination times out
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Handle transient process states during iteration
                    continue
            print(f"Port {port} is free.")


    def _get_running_vllm_model(self, base_url: str) -> Optional[str]:
        """
        Queries the vLLM /v1/models endpoint to check which model is loaded.
        Returns the model ID string if running, or None if connection fails.
        """
        try:
            # vLLM is OpenAI compatible, so it returns {"data": [{"id": "model_name", ...}]}
            resp = requests.get(f"{base_url}/v1/models", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["id"]
        except requests.RequestException:
            pass
        return None

    def _wait_for_server(self, url: str, timeout: int = 60) -> bool:
        """
        Health Check: Blocks until server returns 200 OK.
        Bigger models may need longer startup times.
        """
        start_time = time.time()
        print(f"Waiting for local inference server at {url}...")
        while time.time() - start_time < timeout:
            try:
                # vLLM and Ollama usually have /health or /v1/models endpoints
                requests.get(url, timeout=1)
                print(f"...inference server spawned at {url}.")
                return True
            except requests.ConnectionError:
                time.sleep(2)
        return False

    def _spawn_server(self, cmd: list[str], health_check_url: str, install_hint: str) -> None:
        """
        Spawns a server process, waits for it to become healthy, and handles startup errors.
        """
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,  # Suppress server stdout
                stderr=subprocess.DEVNULL   # Suppress server stderr
            )
        except FileNotFoundError:
            raise RuntimeError(install_hint)

        if not self._wait_for_server(health_check_url):
            self.stop_server()
            raise RuntimeError(f"Failed to start server at {health_check_url}.")

    def ensure_vllm(self, model_name: str) -> None:
        """Ensures a vLLM server is running with the specified model."""
        # Check if vLLM is already running the requested model
        if self._get_running_vllm_model(VLLM_BASE_URL) == model_name:
            print(f"vLLM is already serving {model_name}. Connecting...")
            return

        print(f"Switching to vLLM ({model_name}). Terminating processes on Ports {VLLM_PORT} and {OLLAMA_PORT} to free CPU/GPU...")
        self._kill_processes_on_ports(VLLM_PORT, OLLAMA_PORT)

        print(f"Spawning vLLM server for {model_name}...")
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name, "--port", str(VLLM_PORT),
            "--trust-remote-code", "--gpu-memory-utilization", str(VLLM_GPU_UTIL)
        ]
        vllm_health_url = f"http://localhost:{VLLM_PORT}/v1/models"
        self._spawn_server(vllm_cmd, vllm_health_url, "vLLM is not installed. Install via:  pip install vllm")

    def ensure_ollama(self) -> None:
        """Ensures an Ollama server is running."""
        # 1. Check ports
        ollama_open = self._is_port_open(OLLAMA_PORT)
        vllm_open = self._is_port_open(VLLM_PORT)

        # 2. Fast Path
        if ollama_open:
            if vllm_open:
                self._kill_processes_on_ports(VLLM_PORT)
            return

        # 3. Startup: If Ollama wasn't open, start it.
        if not ollama_open:
            # Potential zombie process cleanup
            self._kill_processes_on_ports(OLLAMA_PORT)

            print("Spawning Ollama server process...")
            ollama_cmd = ["ollama", "serve"]
            ollama_health_url = f"http://localhost:{OLLAMA_PORT}"
            self._spawn_server(ollama_cmd, ollama_health_url, "Ollama not found. Install via 'curl -fsSL https://ollama.com/install.sh | sh'") # noqa

# ----------------------------------- Client ---------------------------------- #

class LLMClient:
    """
    LLM Client supporting
       - Open source: vLLM / Ollama / Huggingface with local CPU/GPU inference.
       - Commercial: OpenAI, Gemini, Anthropic, etc - any provider supported by LiteLLM.

    Supports:
        - stateless & stateful interactions
        - streaming & non-streaming responses
        - text-only & multimodal inputs
        - images provided as
            (1) file paths or
            (2) raw bytes
    """

    def __init__(self) -> None:

        self.messages: List[Tuple[str, str]] = [] # [role, message] - only store text for efficiency
        self.sys_prompt = ""
        self.server_manager = LocalServerManager()

    # ----------------------------------- Data Wrangling ---------------------------------- #
    def _process_image(self, img: Union[Path, bytes, str]) -> str:
        """Standardizes image input to Base64 string."""
        if isinstance(img, str) and img.startswith("data:image"):
            return img # Already base64

        if isinstance(img, (Path, str)):
            path = Path(img)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            with open(path, "rb") as img_file:
                data = img_file.read()
                mime_type = f"image/{path.suffix[1:].lower()}"
                if mime_type == "image/jpg": mime_type = "image/jpeg"

        elif isinstance(img, bytes):
            data = img
            if data.startswith(b'\xff\xd8'): mime_type = "image/jpeg"
            elif data.startswith(b'\x89PNG'): mime_type = "image/png"
            elif data.startswith(b'GIF8'): mime_type = "image/gif"
            elif data.startswith(b'RIFF'): mime_type = "image/webp"
            else: mime_type = "image/jpeg" # Default fallback

        else:
            raise ValueError("Unsupported image type")

        encoded_string = base64.b64encode(data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

    def _resolve_routing(self, model_input: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Parses 'provider/model' and handles local server spawning.
        Returns: (litellm_model_string, api_base, custom_llm_provider)
        """
        if "/" not in model_input:
            raise ValueError("Model must be in format 'provider/model_name' (e.g. 'openai/gpt-4' or 'ollama/llama3')")

        provider, model_name = model_input.split("/", 1)

        api_base = None
        custom_llm_provider = None

        if provider == "hosted_vllm":
            self.server_manager.ensure_vllm(model_name)
            api_base = f"http://localhost:{VLLM_PORT}/v1"
            custom_llm_provider = "hosted_vllm"
        elif provider == "ollama":
            self.server_manager.ensure_ollama()
            api_base = f"http://localhost:{OLLAMA_PORT}"
            custom_llm_provider = "ollama"

        # If provider is commercial (openai, anthropic, etc), LiteLLM handles it natively.
        # We just pass the original string (e.g., "anthropic/claude-3-opus")

        return model_input, api_base, custom_llm_provider

    # -------------------------------- Core LLM Interaction -------------------------------- #
    def get_embedding(
        self, model: str,
        input_text: Union[str, List[str]],
        **model_kwargs: Dict[str, any]
    ) -> EmbeddingResponse:

        # Resolve routing (for local inference: starts server)
        final_model, api_base, custom_llm_provider = self._resolve_routing(model)
        model_kwargs = {"extra_body": model_kwargs} if custom_llm_provider else model_kwargs

        response = embedding(
            model=final_model,
            input=input_text,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider, # defaults to None for commercial providers
            **model_kwargs
        )
        return response

    def api_query(self,
        model: str,
        user_msg: Optional[str] = None,
        user_msg_history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = True,
        **kwargs: Dict[str, any]
    ) -> Iterator[str] | ChatCompletion:

        img_data = []
        if img is not None:
            items = img if isinstance(img, (list, tuple)) else [img]
            img_data = [self._process_image(i) for i in items]

        # 1. Prepare Messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_msg_history:
            for msg in user_msg_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        if user_msg or img:
            content_payload = []
            if user_msg is not None:
                # Text-only Format: Simple string
                content_payload.append({"type": "text", "text": user_msg})
            if img is not None:
                # Multimodal Format: List of dictionaries
                for img_b64 in img_data:
                    content_payload.append({"type": "image_url", "image_url": {"url":img_b64}})

            messages.append({"role": "user", "content": content_payload})

        # 2. Resolve Routing (for local inference: starts server)
        start_time = time.time()
        final_model, api_base, custom_llm_provider = self._resolve_routing(model)
        elapsed = time.time() - start_time
        print(f"Model routing resolved in {elapsed:.2f} seconds. Using model: {final_model}")

        # 3. Execute request via LiteLLM
        try:
            response = completion(
                model=final_model,
                messages=messages,
                stream=stream,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider, # defaults to None for commercial providers
                **kwargs # Pass temperature, top_p, etc. - supports model-specific params
            )

            if stream is False:
                return response
            else:
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        try:
                            yield content
                        except GeneratorExit:
                            return

        except Exception as e:
            if stream:
                yield f"API Error: {e!s}"
            raise e


    def chat(self, model: str,
        user_msg: str,
        system_prompt: Optional[str] = "",
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = True,
        **kwargs: Dict[str, any]
    ) -> Iterator[str]|ChatCompletion:
        """Stateful Chat Wrapper around api_query to maintain conversation history."""

        api_response = self.api_query(
            model=model,
            user_msg=user_msg,
            user_msg_history=self.messages,
            system_prompt=system_prompt,
            img=img,
            **kwargs)

        if stream is False:
            response = api_response
            self.messages.append({"role": "user", "content": user_msg})
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response

        else:
            response = ""
            for chunk in api_response:
                response += chunk
                try:
                    yield chunk
                except GeneratorExit:
                    return

            self.messages.append({"role": "user", "content": user_msg})
            self.messages.append({"role": "assistant", "content": response})

    # ----------------------------------- Cleanup ---------------------------------- #
    def close(self) -> None:
        """Cleans up any background processes."""
        self.server_manager.stop_server()
