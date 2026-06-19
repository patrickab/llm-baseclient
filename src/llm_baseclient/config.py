import os
from pathlib import Path
import subprocess
from typing import TypedDict

OLLAMA_PORT = 11434

TABBY_DIR = os.path.join(os.path.expanduser("~"), "tabbyAPI")
TABBY_PORT = 5000

VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"
VLLM_GPU_UTIL = 0.8  # Limit vLLM to 0,x% VRAM so local machine doesn't freeze

MAX_PARALLEL_REQUESTS = 100  # For batch processing - adjust to system/provider limits
MAX_TOKEN_DEFAULT_VLLM = 8192  # Maximum context length - lower/increase to your GPU/CPU/Model requirements


# --- Model Configuration Types ---
class VllmConfig(TypedDict, total=False):
    vllm_cmd: str


class ExllamaConfig(TypedDict):
    max_seq_len: int
    cache_mode: str


class ModelConfigs(TypedDict, total=False):
    vllm: dict[str, VllmConfig]
    exllama: dict[str, ExllamaConfig]


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

# --- Dynamic Discovery: Ollama ---
MODELS_OLLAMA: list[str] = []
MODELS_OLLAMA_EXCLUDED = ["bge-m3", "nomic-embed-text"]
try:
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    lines = res.stdout.splitlines()[1:]
    models = (line.split()[0] for line in lines)
    MODELS_OLLAMA = [f"ollama/{model}" for model in models if not any(excluded in model for excluded in MODELS_OLLAMA_EXCLUDED)]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass

# --- Dynamic Discovery: VLLM (HuggingFace) ---
_HUGGINGFACE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MODELS_VLLM: list[str] = (
    [
        f"hosted_vllm/{m.name.replace('models--', '', 1).replace('--', '/', 1)}"
        for m in _HUGGINGFACE_DIR.iterdir()
        if m.name.startswith("models--")
    ]
    if _HUGGINGFACE_DIR.exists()
    else []
)

# --- Dynamic Discovery: TabbyAPI ---
_tabby_models_dir = Path.home() / "tabbyAPI" / "models"
MODELS_EXLLAMA: list[str] = (
    [f"tabby/{m.name}" for m in _tabby_models_dir.iterdir() if m.name != "place_your_models_here.txt"]
    if _tabby_models_dir.exists()
    else []
)

AVAILABLE_MODELS = MODELS_GEMINI + MODELS_OPENAI + MODELS_OLLAMA + MODELS_VLLM + MODELS_EXLLAMA


SYS_NOTE_TO_OBSIDIAN_YAML = """
  Your task is to take a user's notes and convert them into a structured YAML format suitable for Obsidian.

  # **Instructions**:
  - **Aliases**: Include common synonyms, abbreviations, alternative phrasings.
  - **Tags**: Include 1-5 general topic keywords. When selecting tags, prioritize consistency:
      - Order tags by relevance to the main topic.
      - Use tags that notes on related topics would likely have (lower case with - separator).
      - Try to add as many relevant tags as possible.
      - Avoid overly specific or unique tags that dont help cluster notes.
  - **Summary**: Concise, one-line summary suitable for hover preview or search.
  - **Format**: Return a **raw YAML header** only. Do not include backticks, code fences, or extra formatting.

  **Output format**:
    ---
    title: {{file_name_no_ext}}
    aliases: [abbreviation, synonym_1, <...>, synonym_n] # 1-4 alternate names
    tags: [domain_1, ..., domain_n] # 1-6 related keywords
    summary: ""
    ---
"""
