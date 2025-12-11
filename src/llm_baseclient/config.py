OLLAMA_PORT = 11434

VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"
VLLM_GPU_UTIL = 0.7 # Limit vLLM to 0,x% VRAM so local machine doesn't freeze
