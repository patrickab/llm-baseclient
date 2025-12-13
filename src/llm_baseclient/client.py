"""
Swiss Army Knife LLM client.

Supports +2100 LLMs from +100 providers via LiteLLM.
Supports hardware-aware GPU/CPU local inference (vLLM/Ollama).
Supports multimodal inputs (images via paths, bytes, data URIs, URLs) - voice coming soon!
"""

import base64
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import filetype
from litellm import completion, embedding
from litellm.utils import EmbeddingResponse
from openai.types.chat import ChatCompletion

from llm_baseclient.config import OLLAMA_PORT, VLLM_PORT
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
        mime_type = kind.mime if kind else "image/jpeg"
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
            self.server_manager.ensure_ollama()
            return model_input, f"http://localhost:{OLLAMA_PORT}", "ollama"

        # For commercial providers (e.g., openai, anthropic), LiteLLM handles routing natively.
        # api_base and custom_llm_provider remain None, and the original model string is used.

        return model_input, None, None

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

        img_data = []
        if img is not None:
            items = img if isinstance(img, (list, tuple)) else [img]
            img_data = [self._process_image(i) for i in items]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_msg_history:
            messages.extend(user_msg_history)

        if user_msg or img_data:
            content_payload = [{"type": "text", "text": user_msg}] if user_msg else []
            content_payload.extend([{"type": "image_url", "image_url": {"url": i}} for i in img_data])
            messages.append({"role": "user", "content": content_payload})

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


if __name__ == "__main__":
    client = LLMClient()

    """Stateless API call (non-streaming)"""
    # ------------------- Commercial Providers ------------------- #
    # Assumes API keys set in environment variables using provider conventions.
    # response = client.api_query(model="openai/gpt-5.2", user_msg="Hello, world!")
    # response = client.api_query(model="gemini/gemini-3-pro", user_msg="Hello, world!")

    # --------------- Open-Source / Local Providers -------------- #
    # Assumes vLLM is installed - automatically downloads model if not present.
    response = client.api_query(model="hosted_vllm/Qwen/Qwen3-0.6B", user_msg="Hello, world!")
    # Assumes Ollama is installed - assumes model is already downloaded.
    response = client.api_query(model="ollama/gemma3:4b", user_msg="Hello, world!")

    # -------------------- Responses Handling -------------------- #
    # Responses are ChatCompletion objects
    response = response.choices[0].message.content
    logger.info("Stateless Response:\n" + response)

    """Stateless API call (streaming)"""
    # Assumes Ollama is installed - assumes model is already downloaded.
    stream = client.api_query(model="ollama/gemma3:4b", user_msg="Tell me a joke!", stream=True)
    response = ""
    logger.info("Streaming response:")
    for chunk in stream:
        print(chunk, end="", flush=True)
        response += chunk

    print("\n\n")

    """Stateful chat interactions"""
    response = client.chat(model="ollama/gemma3:4b", user_msg="Tell me a joke.", stream=False)
    response = client.chat(model="ollama/gemma3:4b", user_msg="Not funny - another one.", stream=False)
    logger.info("Message History:\n\n" + str(client.messages))

    """Image input examples"""
    # Adjust model as needed - Ministral-3 runs fast on weak laptop CPUs
    visual_model = "ollama/ministral-3:3b"

    def print_stream(stream: Iterator[str]) -> None:
        print("\nImage Input Response:")
        for chunk in stream:
            print(chunk, end="", flush=True)
        print("\n\n")

    # Image via web URL
    logger.info("Image via web URL")
    stream = client.api_query(
        model=visual_model,
        user_msg="Should I take this fight?",
        img="https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest/scale-to-width-down/256?cb=20220523172438",
        stream=True,
    )
    print_stream(stream)

    # Image via local file path
    logger.info("Image via local file path")
    stream = client.api_query(
        model=visual_model, user_msg="Should I take this fight?", img=Path("./assets/example-img.webp"), stream=True
    )
    print_stream(stream)

    # Supports different file formats & does not require Path input
    logger.info("Image via local JPEG file path")
    stream = client.api_query(
        model=visual_model, user_msg="Analyze why this picture is funny.", img="./assets/example-img.jpeg", stream=True
    )
    print_stream(stream)

    # Supports multiple images per request (as list of Paths, bytes, NOT mixed)
    logger.info("Multiple images via local file paths")
    stream = client.api_query(
        model=visual_model,
        user_msg="Describe both images.",
        img=[Path("./assets/example-img.webp"), Path("./assets/example-img.jpeg")],
        stream=True,
    )
    print_stream(stream)

    # Supports raw bytes input
    logger.info("Image via raw bytes")
    with open("./assets/example-img.jpeg", "rb") as f:
        img_bytes = f.read()

    stream = client.api_query(model=visual_model, user_msg="What is in this image?", img=img_bytes, stream=True)
    print_stream(stream)

    # Cleanup explicitly
    client.kill_inference_engines()
