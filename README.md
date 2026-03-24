# llm-baseclient

A Python LLM client based on [LiteLLM](https://www.litellm.ai/) with multimodal support that unifies access to 2100+ models from 100+ providers, with automatic CPU/GPU local inference server management.

Exposes vLLM/Ollama/OpenAI/Claude/Gemini as simplified Python API.

## Use Cases

- **Multi-provider evaluation**: Benchmark same prompt across OpenAI, Anthropic, open-source models without rewriting logic
- **Image analysis at scale**: Process hundreds of images in parallel with automatic VRAM management

## Usage

**Route requests transparently across providers:**

```python
from llm_baseclient import LLMClient

client = LLMClient()

# Same API, different backends
response = client.api_query(model="openai/gpt-4", user_msg="hello")
response = client.api_query(model="ollama/llama2:7b", user_msg="hello")
response = client.api_query(model="hosted_vllm/meta-llama/Llama-2-7b-hf", user_msg="hello")

# For VLLM you can use ANY available Huggingface LLM ID with `hosted_vllm` prefix.
# LLM-Baseclient will automatically download & serve this model on your local GPU.
# Startup behavior uses model defaults but can be customized (refer to src/examples)
```

**Process images from paths, bytes, URLs, or data URIs:**
```python
# All of these work identically
client.api_query(model="vision_model", user_msg="describe this", img="/path/to/image.jpg")
client.api_query(model="vision_model", user_msg="describe this", img=image_bytes)
client.api_query(model="vision_model", user_msg="describe this", img="https://example.com/img.jpg")
```

**Run batch inference in parallel:**
```python
batch = [
    {"user_msg": "Summarize this", "img": "image1.jpg"},
    {"user_msg": "Compare these", "img": ["image2.jpg", "image3.jpg"]},
    {"system_prompt": "Output JSON", "user_msg": "Extract data", "img": "image4.jpg"},
]

results = client.batch_api_query(requests=batch, model="vision_model", max_workers=8)
# Returns list of responses (or exceptions per request)
```

**Maintain conversation state automatically:**
```python
response = client.chat(model="ollama/gemma3:4b", user_msg="Tell me a joke")
response = client.chat(model="ollama/gemma3:4b", user_msg="Tell it again but shorter")
# Message history maintained automatically in client.messages
```

**Generate embeddings for RAG workflows:**
```python
embeddings = client.get_embedding(
    model="ollama/embeddinggemma:300m",
    input_text=["document 1", "document 2"]
)
```

**Further remarks**

Model identifiers must match [LiteLLM Model Specification](https://models.litellm.ai/) convention.
See `src/examples/llm-client.py` for detailled usage examples with explanations.

## Real Limitations

**Single-GPU, sequential model switching**: This isn't a multi-tenant inference server. If you switch from Llama to Mistral, the library kills the old process and spawns a new one. Designed for single-user, single-GPU setups.

**No built-in retry/fallback**: If your primary provider fails, there's no automatic failover to a backup. You handle that logic.

**Process-level server management**: vLLM and Ollama servers run as separate processes. No containerization, Docker, or distributed setup out of the box.


## Quick Start

```python
from llm_baseclient import LLMClient

client = LLMClient()

# Commercial API (set OPENAI_API_KEY env var)
response = client.api_query(
    model="openai/gpt-4",
    user_msg="Why do italians hate pinapple pizza?",
    temperature=0.7,
    max_tokens=200
)
print(response.choices[0].message.content)

# Local model (auto-spawns vLLM server)
response = client.api_query(
    model="hosted_vllm/Qwen/Qwen3-0.6B",
    user_msg="Why do italians hate pinapple pizza?",
)
print(response.choices[0].message.content)

# Cleanup when done (kills background servers)
client.kill_inference_engines()
```

## Architecture

- **`client.py`**: Routing logic, message construction, API layer
- **`local_server_manager.py`**: vLLM/Ollama lifecycle (spawn, health-check, graceful shutdown)
- **`config.py`**: Port assignments, GPU memory tuning, token limits
- **`logger.py`**: Structured logging with Rich

The library wraps LiteLLM for actual API calls, adding local server automation and multimodal handling on top.

## Scope

This is a **client library**, not a deployment framework. It assumes:
- Local installation of vLLM/Ollama, or API credentials for cloud providers
- Authentication via env vars following LiteLLM conventions: (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- GPU memory and model downloading are pre-configured
