OLLAMA_PORT = 11434

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