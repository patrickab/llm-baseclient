import os
from pathlib import Path
import subprocess
from typing import TypedDict

OLLAMA_PORT = 11434

VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"
VLLM_GPU_UTIL = 0.8  # Limit vLLM to 0,x% VRAM so local machine doesn't freeze

MAX_PARALLEL_REQUESTS = 100  # For batch processing - adjust to system/provider limits
MAX_TOKEN_DEFAULT_VLLM = 8192  # Maximum context length - lower/increase to your GPU/CPU/Model requirements


# --- Model Configuration Types ---
class VllmConfig(TypedDict, total=False):
    vllm_cmd: str


class ModelConfigs(TypedDict, total=False):
    vllm: dict[str, VllmConfig]


def vllm_default_command(model_name: str) -> list[str]:
    return [
        "vllm",
        "serve",
        model_name,
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        str(VLLM_GPU_UTIL),
        "--max-model-len",
        str(MAX_TOKEN_DEFAULT_VLLM),
    ]


# --- Static Model Definitions ---
MODELS_GEMINI = (
    [
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.1-pro-preview",
    ]
    if os.getenv("GEMINI_API_KEY")
    else []
)

MODELS_OPENAI = (
    [
        "openai/gpt-5.1",
        "openai/o1",
        "openai/gpt-5-mini",
        "openai/gpt-4o",
    ]
    if os.getenv("OPENAI_API_KEY")
    else []
)

MODELS_DEEPSEEK = (
    [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    ]
    if os.getenv("DEEPSEEK_API_KEY")
    else []
)

MODELS_OLLAMA_EXCLUDED = ["bge-m3", "nomic-embed-text"]


def discover_ollama_models() -> list[str]:
    """Lists locally installed Ollama models (excluding embedding models)."""
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    lines = res.stdout.splitlines()[1:]
    models = (line.split()[0] for line in lines if line.strip())
    return [f"ollama/{model}" for model in models if not any(excluded in model for excluded in MODELS_OLLAMA_EXCLUDED)]


def discover_vllm_models() -> list[str]:
    """Lists models in the local HuggingFace cache, usable via vLLM."""
    hf_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_dir.exists():
        return []
    return [
        f"hosted_vllm/{m.name.replace('models--', '', 1).replace('--', '/', 1)}"
        for m in hf_dir.iterdir()
        if m.name.startswith("models--")
    ]


AVAILABLE_MODELS = MODELS_GEMINI + MODELS_OPENAI + MODELS_DEEPSEEK + discover_ollama_models() + discover_vllm_models()
