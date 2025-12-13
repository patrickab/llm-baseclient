"""
Swiss Army Knife LLM client.

Supports +2100 LLMs from +100 providers via LiteLLM.
Supports hardware-aware GPU/CPU local inference (vLLM/Ollama).
Supports multimodal inputs (images via paths, bytes, data URIs, URLs) - voice coming soon!
"""

import base64
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import filetype
from litellm import batch_completion, completion, embedding
from litellm.types.utils import ModelResponse
from litellm.utils import EmbeddingResponse
from openai.types.chat import ChatCompletion

from llm_baseclient.config import MAX_PARALLEL_REQUESTS, OLLAMA_PORT, VLLM_PORT
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

    def __init__(self) -> None:
        """
        Initializes the LLMClient with an empty message history, an empty system prompt,
        and a _LocalServerManager for handling local inference servers.
        """
        # Stores conversation history as a list of (role, message) tuples.
        # Only text content is stored for efficiency; multimodal inputs are processed on-the-fly.
        self.messages: List[Dict[str, str]] = []
        self.server_manager = _LocalServerManager()

    # ----------------------------------- Data Wrangling ---------------------------------- #
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

    def _resolve_routing(self, model_input: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Parses 'provider/model' and handles local server spawning."""
        if "/" not in model_input:
            raise ValueError("Model format must be 'provider/model_name'")

        provider, model_name = model_input.split("/", 1)

        if provider == "hosted_vllm":
            self.server_manager.ensure_vllm(model_name)
            return model_input, f"http://localhost:{VLLM_PORT}/v1", "hosted_vllm"
        elif provider == "ollama":
            self.server_manager.ensure_ollama(model_name)
            return model_input, f"http://localhost:{OLLAMA_PORT}", "ollama"

        # For commercial providers (e.g., openai, anthropic), LiteLLM handles routing natively.
        # api_base and custom_llm_provider remain None, and the original model string is used.

        return model_input, None, None

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
    # -------------------------------- Core LLM Interaction -------------------------------- #
    def get_embedding(self, model: str, input_text: Union[str, List[str]], **model_kwargs: Dict[str, any]) -> EmbeddingResponse:  # type: ignore
        """
        Generates embeddings for the given input text using the specified model.
        Handles routing for local inference servers (vLLM, Ollama) and commercial providers.

        Args:
            model: Model identifier in [LiteLLM Format](https://models.litellm.ai/) - e.g., 'openai/text-embedding-ada-002').
            input_text: The text or list of texts to embed.
            **model_kwargs: Additional keyword arguments passed directly to the LiteLLM `embedding` call.

        Returns:
            An EmbeddingResponse object containing the generated embeddings.
        """
        final_model, api_base, custom_llm_provider = self._resolve_routing(model)
        # For custom local providers, model_kwargs need to be nested under 'extra_body'.
        model_kwargs = {"extra_body": model_kwargs} if custom_llm_provider else model_kwargs

        response = embedding(
            model=final_model, input=input_text, api_base=api_base, custom_llm_provider=custom_llm_provider, **model_kwargs
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
        **kwargs: Dict[str, any],
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

        final_model, api_base, custom_llm_provider = self._resolve_routing(model)
        try:
            response = completion(
                model=final_model,
                messages=messages,
                stream=stream,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,  # Defaults to None for commercial providers.
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
                    for chunk in response:
                        # Extract content from streaming chunks.
                        content = chunk.choices[0].delta.content
                        if content:
                            try:
                                yield content
                            except GeneratorExit:
                                return

                return stream_generator()
        except Exception as e:
            raise e

    def batch_api_query(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        **kwargs: Dict[str, any],
    ) -> List[Union[ModelResponse, Exception]]:
        """
        Executes parallel stateless batch requests with high throughput.

        Args:
            requests: A list of dictionaries. Each dict must contain keys corresponding
                      to arguments of `_construct_message_payload`
                      (e.g., {'user_msg': '...', 'img': ...}).
            model: The model identifier.
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

        # 2. Execute Parallel Batch (IO-bound)
        final_model, api_base, custom_llm_provider = self._resolve_routing(model)

        responses = batch_completion(
            model=final_model,
            messages=messages_batch,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            max_workers=MAX_PARALLEL_REQUESTS,
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
        **kwargs: Dict[str, any],
    ) -> Iterator[str] | ChatCompletion:
        """
        Stateful chat wrapper around `api_query` to maintain and update conversation history.

        Args:
            model: The model identifier in 'provider/model_name' format.
            user_msg: The current user message.
            system_prompt: An optional system-level instruction for the model.
            img: Optional image input(s) for multimodal messages.
            stream: If True, streams the response. If False, returns the complete response.
            **kwargs: Additional keyword arguments passed to `api_query`.

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
            **kwargs,
        )

        if stream is False:
            self.messages.append({"role": "user", "content": user_msg})
            self.messages.append({"role": "assistant", "content": api_response.choices[0].message.content})
            return api_response

        def _chat_generator() -> Iterator[str]:
            """
            Generator wrapper to isolate `yield` keyword from outer function scope.
            Allows the outer function to conditionally return `Iterator[Str] | ChatCompletion`
            """
            full_response_text = ""
            for chunk in api_response:
                full_response_text += chunk
                yield chunk

            # Update history after stream finishes
            self.messages.append({"role": "user", "content": user_msg})
            self.messages.append({"role": "assistant", "content": full_response_text})

        return _chat_generator()

    # ----------------------------------- Cleanup ---------------------------------- #
    def kill_inference_engines(self) -> None:
        """Cleans up any background processes."""
        self.server_manager._kill_inference_engines(targets={"vllm", "ollama", "ollama runner"})
