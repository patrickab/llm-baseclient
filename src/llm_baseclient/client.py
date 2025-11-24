import base64
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Iterator, List, Optional, Tuple

from litellm import completion
from openai.types.chat import ChatCompletion

from .config import MODELS_ANTHROPIC, MODELS_GEMINI, MODELS_OPENAI


def _has_nvidia_gpu() -> bool:
    """
    Check for NVIDIA GPU availability using standard library only.
    Returns True if 'nvidia-smi' is found and executes successfully.
    """
    # 1. Check if the binary exists in PATH
    if not shutil.which("nvidia-smi"):
        return False
            
    # 2. Try executing it to ensure drivers are actually working
    try:
        # Run nvidia-smi to query GPU count. 
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], 
            capture_output=True, 
            text=True, 
            timeout=3
        )

        if result.returncode == 0:
            count = int(result.stdout.strip())
            return count > 0

    except (subprocess.SubprocessError, ValueError):
        return False

    return False

class LLMClient:

    def __init__(self) -> None:

        self.messages: List[Tuple[str, str]] = [] # [role, message] - only store text for efficiency
        self.sys_prompt = ""

    # ----------------------------------- Data Wrangling ---------------------------------- #

    def _img_path_to_base64(self, img_path: Path) -> str:
        """Convert image file to base64 string."""
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/{img_path.suffix[1:]};base64,{encoded_string}"

    def _img_bytes_to_base64(self, img_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        # Detect image format from magic bytes
        if img_bytes.startswith(b'\xff\xd8\xff'):
            img_format = 'jpeg'
        elif img_bytes.startswith(b'\x89PNG'):
            img_format = 'png'
        elif img_bytes.startswith(b'GIF8'):
            img_format = 'gif'
        elif img_bytes.startswith(b'RIFF') and img_bytes[8:12] == b'WEBP':
            img_format = 'webp'
        elif img_bytes.startswith(b'BM'):
            img_format = 'bmp'
        else:
            raise ValueError("Unsupported image format")

        encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{img_format};base64,{encoded_string}"

    # -------------------------------- Core LLM Interaction -------------------------------- #

    def api_query(
        self, model: str,
        user_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = True,
        **kwargs: Dict[str, any]
    ) -> Iterator[str] | ChatCompletion:
        """
        Stateless API call using LiteLLM to unify the request format.
        Routes to the correct local/cloud API client.

        For local models:
            - GPU: assumes vLLM
            - CPU: assumes Ollama
        """
        if not isinstance(img, (Path, bytes, List[Path], List[bytes], type(None))):
            raise ValueError("img must be None, Path or bytes")

        if isinstance(img, Path):
            img_data = self._img_path_to_base64(img)
            img_data = [img_data]
        elif isinstance(img, bytes):
            img_data = self._img_bytes_to_base64(img)
            img_data = [img_data]
        elif isinstance(img, list):
            img_data = []
            for item in img:
                if isinstance(item, Path):
                    img_data.append(self._img_path_to_base64(item))
                elif isinstance(item, bytes):
                    img_data.append(self._img_bytes_to_base64(item))
                else:
                    raise ValueError("img list items must be of type Path or bytes")

        # 1. Set defaults
        api_base = None
        custom_llm_provider = None

        # 2. Prepare Messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.messages:
            messages.extend(self.messages)
        if user_message or img:
            content_payload = []
            if user_message is not None:
                # Text-only Format: Simple string
                content_payload.append({"type": "text", "text": user_message})
            if img is not None:
                # Multimodal Format: List of dictionaries
                for img_b64 in img_data:
                    content_payload.append({"type": "image", "image_data": img_b64})

            messages.append({"role": "user", "content": content_payload})

        # 3. Determine Provider
        is_cloud_model = (model in MODELS_OPENAI) or (model in MODELS_GEMINI) or (model in MODELS_ANTHROPIC)

        if not is_cloud_model:
            # Routing Logic: Prefer vLLM (GPU) > Ollama (CPU)
            if _has_nvidia_gpu():
                # Use vLLM settings
                api_base = str(self.vllm_client.base_url)
                custom_llm_provider = "openai" # vLLM mimics OpenAI

            else:
                # Use Ollama settings
                api_base = "http://localhost:11434"
                custom_llm_provider = "ollama"

                # LiteLLM requires 'ollama/' prefix
                if not model.startswith("ollama/"):
                    model = f"ollama/{model}"
        else:
            if model in MODELS_OPENAI:
                model = f"openai/{model}"
            elif model in MODELS_GEMINI:
                model = f"gemini/{model}"
            elif model in MODELS_ANTHROPIC:
                model = f"anthropic/{model}"

        # 4. Execute request via LiteLLM
        try:
            assert messages[0]['role'] == 'system'
            response = completion(
                model=model,
                messages=messages,
                stream=stream,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                **kwargs # Pass temperature, top_p, etc.
            )

            if stream is False:
                return response
            else:
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        except Exception as e:
            yield f"API Error: {e!s}"

    def chat(self, model: str,
        user_message: str,
        system_prompt: Optional[str] = "",
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = True,
        **kwargs: Dict[str, any]
    ) -> Iterator[str]|ChatCompletion:
        """
        Stateful Chat Wrapper
        Routes to the correct local/cloud API client.

        For local models:
            - GPU: assumes vLLM
            - CPU: assumes Ollama
        """

        api_response = self.api_query(
            model=model,
            user_message=user_message,
            system_prompt=system_prompt,
            img=img,
            **kwargs)

        if stream is False:
            response = api_response
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response
        else:
            response = ""
            for chunk in api_response:
                response += chunk
                yield chunk
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": response})
