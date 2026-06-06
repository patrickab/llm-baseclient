"""
Swiss Army Knife LLM client.

Supports +2100 LLMs from +100 providers via LiteLLM.
Supports hardware-aware GPU/CPU local inference (vLLM/Ollama).
Supports multimodal inputs (images via paths, bytes, data URIs, URLs) - voice coming soon!
"""

import base64
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import filetype
from litellm import batch_completion, completion, embedding
from litellm.types.utils import ModelResponse
from litellm.utils import EmbeddingResponse
from openai.types.chat import ChatCompletion
from PIL import Image
import requests

from llm_baseclient.config import MAX_PARALLEL_REQUESTS, OLLAMA_PORT, SYS_NOTE_TO_OBSIDIAN_YAML, TABBY_PORT, VLLM_PORT, ModelConfigs
from llm_baseclient.local_server_manager import _LocalServerManager
from llm_baseclient.logger import get_logger

logger = get_logger()


# ----------------------------------- Client ---------------------------------- #
class LLMClient:
    """
    LLM Client supporting
       - Open source: vLLM / Ollama / Huggingface with local CPU/GPU inference.
       - Commercial: OpenAI, Gemini, Anthropic, etc - any provider supported by LiteLLM.

    Automatically spawns & manages local inference servers (vLLM, Ollama).
    Allowing dynamic switching between backends/models during runtime.

    Assumes:
        For Commercial Providers:
            - API keys in environment.
        For vLLM / Ollama:
            - Local provider software installed.
            - Local models already downloaded.

    Supports:
        - stateless & stateful interactions
        - streaming & non-streaming responses
        - text-only & multimodal inputs
        - images provided as
            (1) file paths
            (2) raw bytes
            (3) base64 data URIs
            (4) web URLs
    """

    def __init__(self, model_configs: ModelConfigs | None = None) -> None:
        """
        Initializes the LLMClient with an empty message history, an empty system prompt,
        and a _LocalServerManager for handling local inference servers.

        Args:
            model_configs: Optional per-model configurations for vLLM and ExLlama backends.
        """
        self.messages: List[Dict[str, str]] = []
        self.model_configs = model_configs or {}
        self.server_manager = _LocalServerManager()

    # ----------------------------------- Data Wrangling ---------------------------------- #
    def _lookup_model_config(self, model: str) -> tuple[str | None, dict[str, Any] | None]:
        """Returns (vllm_cmd_list, tabby_config) from stored model_configs for the given model."""
        vcc = self.model_configs.get("vllm", {})
        ecc = self.model_configs.get("exllama", {})
        vllm_cmd = vcc.get(model, {}).get("vllm_cmd")
        tabby_config = ecc.get(model)
        return vllm_cmd.split() if vllm_cmd else None, tabby_config

    def _process_image(self, img: Union[Path, bytes, str]) -> str:
        """Standardizes image inputs (Path, bytes, or data-URI) into a Base64 data URI or passes URL."""

        if isinstance(img, str) and img.startswith(("http://", "https://", "data:image")):
            # LiteLLM/OpenAI can download the image themselves.
            # data:image implies already data URI format.
            return img

        # Normalize to bytes
        if isinstance(img, (Path, str)):
            path = Path(img)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            img = path.read_bytes()

        # Detect mime and encode
        kind = filetype.guess(img)
        mime_type = kind.mime if kind else "image/jpeg"  # fallback to jpeg
        b64_encoded = base64.b64encode(img).decode("utf-8")
        return f"data:{mime_type};base64,{b64_encoded}"

    def _resolve_routing(
        self, model_input: str, vllm_cmd: Optional[str] = None, tabby_config: Optional[dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses 'provider/model' and handles local server spawning.
        vllm_cmd allows customizing vLLM server startup behavior.
        """
        if "/" not in model_input:
            raise Warning(f"Model format must be 'provider/model_name'. Got: {model_input}\nFalling back to openai/{model_input}")

        provider, model_name = model_input.split("/", 1)

        if provider == "hosted_vllm":
            self.server_manager.ensure_vllm(model_name, vllm_cmd=vllm_cmd)
            model_id = model_input
            url = f"http://localhost:{VLLM_PORT}/v1/models/{model_name}"
            provider = "hosted_vllm"
        elif provider == "ollama":
            self.server_manager.ensure_ollama(model_name)
            model_id = model_name
            url = f"http://localhost:{OLLAMA_PORT}"
            provider = "ollama"
        elif provider == "tabby":
            self.server_manager.ensure_tabby(model_name, tabby_config=tabby_config)
            model_id = model_input.replace("tabby/", "")
            url = f"http://localhost:{TABBY_PORT}/v1/"
            provider = "openai"  # Tabby uses OpenAI-compatible API.
        else:  # Commercial provider via LiteLLM
            model_id = model_input
            url = None
            provider = None

        return model_id, url, provider

    def _construct_message_payload(
        self,
        user_msg: Optional[str] = None,
        user_msg_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pure function: Converts raw inputs into a LiteLLM-compatible messages list.
        Decouples data prep from execution.
        """
        # 1. Process Images
        img_data = []
        if img is not None:
            items = img if isinstance(img, (list, tuple)) else [img]
            img_data = [self._process_image(i) for i in items]

        # 2. Build Message History
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_msg_history:
            messages.extend(user_msg_history)

        # 3. Construct Current Turn (Multimodal handling)
        if user_msg or img_data:
            if not img_data:
                # Text-only optimization (lower token overhead)
                messages.append({"role": "user", "content": user_msg})
            else:
                # Multimodal payload
                content_payload = []
                if user_msg:
                    content_payload.append({"type": "text", "text": user_msg})

                # Add all images
                content_payload.extend([{"type": "image_url", "image_url": {"url": i}} for i in img_data])
                messages.append({"role": "user", "content": content_payload})

        return messages

    # -------------------------------- Core LLM Interaction -------------------------------- #
    def get_embedding(
        self, model: str, input_text: Union[str, List[str]], vllm_cmd: Optional[str] = None, **model_kwargs: Dict[str, any]
    ) -> EmbeddingResponse:
        """
        Generates embeddings for the given input text using the specified model.
        Handles routing for local inference servers (vLLM, Ollama) and commercial providers.

        Args:
            model: Model identifier in [LiteLLM Format](https://models.litellm.ai/) - e.g., 'openai/text-embedding-ada-002').
            input_text: The text or list of texts to embed.
            vllm_cmd: Customize server startup behavior
            **model_kwargs: Additional keyword arguments passed directly to the LiteLLM `embedding` call.

        Returns:
            An EmbeddingResponse object containing the generated embeddings.
        """
        model_id, api_base, custom_llm_provider = self._resolve_routing(model, vllm_cmd=vllm_cmd)
        # For custom local providers, model_kwargs need to be nested under 'extra_body'.
        model_kwargs = {"extra_body": model_kwargs} if custom_llm_provider else model_kwargs

        response = embedding(
            model=model_id,
            input=input_text,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            **model_kwargs,
        )
        return response

    def api_query(
        self,
        model: str,
        user_msg: Optional[str] = None,
        user_msg_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = False,
        return_usage: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterator[str] | ChatCompletion:
        """
        Executes a raw API query to an LLM, supporting text-only and multimodal inputs,
        streaming and non-streaming responses, and various providers.

        Args:
            model: Model identifier [LiteLLM Format](https://models.litellm.ai/) - e.g., 'openai/text-embedding-ada-002').
            user_msg: The current user message as a string.
            user_msg_history: A list of message dictionaries representing prior conversation turns.
                              Each dictionary should have 'role' and 'content' keys.
            system_prompt: An optional system-level instruction for the model.
            img: Optional image input(s) as file paths, raw bytes, or a list of these.
            stream: If True, returns an iterator yielding chunks of the response.
                    If False, returns a complete ChatCompletion object.
            return_usage: If True and streaming, yields a final dict with
                          ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
                          after the text stream exhausts.
            **kwargs: Additional keyword arguments passed directly to the LiteLLM `completion` call
                      (e.g., `temperature`, `top_p`, `max_tokens`).

        Returns:
            An iterator of strings if `stream` is True, or a ChatCompletion object if `stream` is False.

        Raises:
            Exception: If an error occurs during the API call.
        """

        messages = self._construct_message_payload(
            user_msg=user_msg, user_msg_history=user_msg_history, system_prompt=system_prompt, img=img
        )

        # tabby does not support reasoning
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_effort = None if "tabby/" in model else reasoning_effort

        # dummy api key for openai-compatibility
        # other models use environment variables or no auth
        if "tabby/" in model and "api_key" not in kwargs:
            kwargs["api_key"] = "tabby-dummy-key"

        # Intercept optional kwargs — explicit kwargs override model_configs
        vllm_cmd = kwargs.pop("vllm_cmd", None)
        tabby_config = kwargs.pop("tabby_config", None)
        if not vllm_cmd and not tabby_config:
            vllm_cmd, tabby_config = self._lookup_model_config(model)

        model_id, api_base, custom_llm_provider = self._resolve_routing(model, vllm_cmd=vllm_cmd, tabby_config=tabby_config)
        try:
            response = completion(
                model=model_id,
                messages=messages,
                stream=stream,
                api_base=api_base,
                reasoning_effort=reasoning_effort,
                custom_llm_provider=custom_llm_provider,  # Defaults to None for commercial providers.
                stream_options={"include_usage": True} if stream else None,
                **kwargs,  # Passes additional model parameters like temperature, top_p, max_tokens.
            )
            if stream is False:
                return response
            else:

                def stream_generator() -> Iterator[str]:
                    """
                    Generator wrapper to isolate `yield` keyword from outer function scope.
                    Allows the outer function to conditionally return `Iterator[Str] | ChatCompletion`
                    """
                    in_reasoning = False
                    last_usage = None
                    for chunk in response:
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            last_usage = {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens,
                            }
                            continue

                        delta = chunk.choices[0].delta
                        reasoning = getattr(delta, "reasoning_content", None)
                        content = delta.content

                        if reasoning:
                            if not in_reasoning:
                                yield "<thought>\n"
                                in_reasoning = True
                            yield reasoning

                        if content:
                            if in_reasoning:
                                yield "\n</thought>"
                                in_reasoning = False
                            yield content

                    if in_reasoning:
                        yield "\n</thought>"

                    if return_usage and last_usage:
                        yield last_usage

                return stream_generator()
        except Exception as e:
            logger.error("Error during API query: %s", e, stacklevel=2)
            return Exception("API query failed")

    def batch_api_query(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        **kwargs: Dict[str, Any],
    ) -> List[Union[ModelResponse, Exception]]:
        """
        Executes parallel stateless batch requests with high throughput.

        Args:
            requests: A list of dictionaries. Each dict must contain keys corresponding
                      to arguments of `_construct_message_payload`
                      (e.g., {'user_msg': '...', 'img': ...}).
            model: The model identifier.
            vllm_cmd: Customize server startup behavior
            **kwargs: Global overrides (e.g., temperature=0.7).

        Returns:
            A list of ModelResponse objects (or Exceptions if a specific request failed).
        """
        # 1. Pre-calculate all message payloads (CPU-bound)
        messages_batch = [
            self._construct_message_payload(
                user_msg=req.get("user_msg"),
                user_msg_history=req.get("user_msg_history"),
                system_prompt=req.get("system_prompt"),
                img=req.get("img"),
            )
            for req in requests
        ]

        # Intercept optional kwargs — explicit kwargs override model_configs
        vllm_cmd = kwargs.pop("vllm_cmd", None)
        if not vllm_cmd:
            vllm_cmd, _ = self._lookup_model_config(model)

        # 2. Execute Parallel Batch (IO-bound)
        model_id, api_base, custom_llm_provider = self._resolve_routing(model, vllm_cmd=vllm_cmd)

        responses = batch_completion(
            model=model_id,
            messages=messages_batch,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            max_workers=MAX_PARALLEL_REQUESTS if "max_workers" not in kwargs else kwargs.pop("max_workers"),
            **kwargs,
        )

        return responses

    def chat(
        self,
        model: str,
        user_msg: str,
        system_prompt: Optional[str] = "",
        img: Optional[Path | List[Path] | bytes | List[bytes]] = None,
        stream: bool = True,
        return_usage: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Iterator[str] | ChatCompletion:
        """
        Stateful chat wrapper around `api_query` to maintain and update conversation history.

        Args:
            model: The model identifier in 'provider/model_name' format.
            user_msg: The current user message.
            system_prompt: An optional system-level instruction for the model.
            img: Optional image input(s) for multimodal messages.
            stream: If True, streams the response. If False, returns the complete response.
            return_usage: If True and streaming, passes through the usage dict from
                          ``api_query`` as the final yielded element.
            **kwargs: Additional keyword arguments passed to ``api_query``.

        Returns:
            An iterator of strings if `stream` is True, or a ChatCompletion object if `stream` is False.
        """

        api_response = self.api_query(
            model=model,
            user_msg=user_msg,
            user_msg_history=self.messages,
            system_prompt=system_prompt,
            img=img,
            stream=stream,
            return_usage=return_usage,
            **kwargs,
        )

        if stream is False:
            api_response: ChatCompletion
            msg = api_response.choices[0].message

            self.messages.append({"role": "user", "content": user_msg})
            if msg.tool_calls:
                self.messages.append(msg.model_dump())
            else:
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    content = f"<thought>\n{reasoning}\n</thought>\n{content}"
                self.messages.append({"role": "assistant", "content": content})
            return api_response

        def _chat_generator() -> Iterator[str]:
            """
            Generator wrapper to isolate `yield` keyword from outer function scope.
            Allows the outer function to conditionally return `Iterator[Str] | ChatCompletion`
            """
            full_response_text = ""
            for chunk in api_response:
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    full_response_text += chunk
                    yield chunk

            # Update history after stream finishes
            self.messages.append({"role": "user", "content": user_msg})
            self.messages.append({"role": "assistant", "content": full_response_text})

        return _chat_generator()

    def add_tool_result(self, tool_call_id: str, output: str) -> None:
        """Injects a tool execution result into the history."""
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(output)})

    # ----------------------------------- History Management ---------------------------------- #
    def store_history(self, file_path: Union[str, Path]) -> None:
        """Store message history to filesystem as JSON."""
        Path(file_path).write_text(json.dumps(self.messages, indent=2), encoding="utf-8")

    def load_history(self, file_path: Union[str, Path]) -> None:
        """Load message history from JSON filesystem."""
        if (p := Path(file_path)).exists():
            self.messages = json.loads(p.read_text(encoding="utf-8"))

    def reset_history(self) -> None:
        """Clear the current conversation history."""
        self.messages = []

    # ----------------------------------- Cleanup ---------------------------------- #
    def kill_inference_engines(self) -> None:
        """Cleans up any background processes."""
        self.server_manager._kill_inference_engines(targets={"vllm", "ollama", "ollama runner", "tabby"})

    # ----------------------------------- Image Utilities ---------------------------------- #
    @staticmethod
    def write_to_md(file_path: Path, message: str) -> None:
        """Writes a simple text message to a markdown file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(message, encoding="utf-8")

    def write_to_obsidian(
        self,
        file_path: Path,
        message: str,
        model: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Writes a message to a markdown file, prepending an AI-generated YAML frontmatter block suitable for Obsidian.

        Args:
            file_path: The pathlib Path where the markdown file should be written.
            message: The content to write.
            model: The LLM model used for generating the header.
            **kwargs: Additional parameters passed to the model (e.g., temperature).
        """
        system_prompt = SYS_NOTE_TO_OBSIDIAN_YAML.replace("{{file_name_no_ext}}", file_path.stem).replace("{{user_notes}}", message)
        response: ChatCompletion = self.api_query(model=model, user_msg=message, system_prompt=system_prompt, stream=False, **kwargs)
        yaml_header = response.choices[0].message.content

        self.write_to_md(file_path, f"{yaml_header}\n{message}")

    @staticmethod
    def downscale_img(
        img: Union[str, bytes, Path, Image.Image],
        max_tokens: int,
        grid_size: int = 28,
        tokens_per_patch: int = 1,
        row_overhead_tokens: int = 1,
        output_format: str = "JPEG",
        quality: int = 85,
    ) -> str:
        """
        Preserves aspect ratio (area-scaling) and aligns to ViT patches (grid_size) for spatial accuracy.
        Optimizes GPU inference efficiency within token budget.

        JPEG: Fastest TTFT (optimized CPU decoding).
        PNG: Max fidelity (lossless).
        """
        # 1. Normalize Input to PIL Image
        if isinstance(img, Image.Image):
            pass
        elif hasattr(img, "image_data"):  # Streamlit PasteResult (if passed directly)
            img = img.image_data  # type: ignore
            if not isinstance(img, Image.Image):
                raise ValueError("Invalid image type from PasteResult.")
        elif isinstance(img, (str, Path)):
            src = str(img)
            if src.startswith("http"):
                img = Image.open(io.BytesIO(requests.get(src, timeout=10).content))
            elif src.startswith("data:image"):
                img = Image.open(io.BytesIO(base64.b64decode(src.partition(",")[-1])))
            else:
                img = Image.open(Path(src))
        else:
            img = Image.open(io.BytesIO(img) if isinstance(img, bytes) else img)

        # 2. Universal Resizing Logic
        def get_tokens(w: int, h: int) -> int:
            pw, ph = math.ceil(w / grid_size), math.ceil(h / grid_size)
            return (pw * ph * tokens_per_patch) + (ph * row_overhead_tokens)

        w, h = img.size
        curr_tokens = get_tokens(w, h)

        scale = 1.0
        if curr_tokens > max_tokens:
            scale = math.sqrt(max_tokens / curr_tokens)

        # Snap to Grid (Preserving Aspect Ratio)
        fw = max(grid_size, round((w * scale) / grid_size) * grid_size)
        fh = max(grid_size, round((h * scale) / grid_size) * grid_size)

        # Iterative refinement to guarantee budget compliance
        while get_tokens(fw, fh) > max_tokens and (fw > grid_size or fh > grid_size):
            if fw > fh:
                fw -= grid_size
            else:
                fh -= grid_size

        if (fw, fh) != (w, h):
            img = img.resize((fw, fh), Image.Resampling.LANCZOS)

        # 3. Configurable Encoding (Optimized for Localhost)
        if output_format.upper() in ["JPEG", "JPG"]:
            if img.mode != "RGB":
                img = img.convert("RGB")
            save_params = {"quality": quality, "optimize": False}
        elif output_format.upper() == "PNG":
            save_params = {"optimize": False}
        else:
            save_params = {"quality": quality}

        buf = io.BytesIO()
        img.save(buf, format=output_format, **save_params)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{output_format.lower()};base64,{b64}"
