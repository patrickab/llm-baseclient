import os

OLLAMA_PORT = 11434

TABBY_DIR = os.path.join(os.path.expanduser("~"), "tabbyAPI")
TABBY_PORT = 5000

VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"
VLLM_GPU_UTIL = 0.8  # Limit vLLM to 0,x% VRAM so local machine doesn't freeze

MAX_PARALLEL_REQUESTS = 100  # For batch processing - adjust to system/provider limits
MAX_TOKEN_DEFAULT_VLLM = 8192  # Maximum context length - lower/increase to your GPU/CPU/Model requirements


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
