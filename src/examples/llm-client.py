import os
from pathlib import Path

import numpy as np

from llm_baseclient.client import LLMClient
from llm_baseclient.logger import get_logger

logger = get_logger()


def main() -> None:
    client = LLMClient()
    # select your preferred vision model here
    # Ministral-3 is a good lightweight option for weak GPUs
    vision_model = "hosted_vllm/mistralai/Ministral-3-3B-Instruct-2512"
    # If you are using CPU use Ollama as VLM client.
    # vision_model = "ollama/ministral-3:3b"

    """Stateless API call (non-streaming)"""
    logger.info("--- Stateless API call (non-streaming) ---")

    # ------------------- Commercial Providers ------------------- #
    # Assumes API keys set in environment variables using provider conventions.
    response = client.api_query(model="openai/gpt-5.2", user_msg="Hello, world!")
    logger.info(f"OpenAI Response: {response.choices[0].message.content}")

    response = client.api_query(model="gemini/gemini-2.5-flash", user_msg="Hello, world!")
    logger.info(f"Gemini Response: {response.choices[0].message.content}")

    # --------------- Open-Source / Local Providers -------------- #
    # Assumes vLLM is installed - automatically downloads model if not present.
    response = client.api_query(model="hosted_vllm/Qwen/Qwen3-0.6B", user_msg="Hello, world!")
    logger.info(f"Qwen Response: {response.choices[0].message.content}")

    # Assumes Ollama is installed - automatically downloads model if not present.
    response = client.api_query(model="ollama/gemma3:4b", user_msg="Hello, world!")
    logger.info(f"Gemma Response: {response.choices[0].message.content}")

    """Stateless API call (streaming)"""
    logger.info("--- Stateless API call (streaming) ---")
    # Assumes Ollama is installed - assumes model is already downloaded.
    stream = client.api_query(model="ollama/gemma3:4b", user_msg="Tell me a joke!", stream=True)
    response = ""
    logger.info("Streaming response (visual output to stdout):")
    for chunk in stream:
        print(chunk, end="", flush=True)
        response += chunk

    print("\n\n")
    logger.info(f"Full Streamed Response: {response}")

    """Stateful chat interactions"""
    logger.info("--- Stateful chat interactions ---")
    response = client.chat(model="ollama/gemma3:4b", user_msg="Tell me a knock knock joke.", stream=False)
    response = client.chat(model="ollama/gemma3:4b", user_msg="Who's there?", stream=False)
    logger.info("Message History:\n\n" + str(client.messages))

    """
    Image input examples

    Supports images as web URLs, local file paths, raw bytes or byte strings.
    Supports multiple images as list.
    """
    logger.info("--- Image input examples ---")

    def print_stream(stream) -> None:  # noqa
        print("\nImage Input Response:")
        for chunk in stream:
            print(chunk, end="", flush=True)
        print("\n\n")

    # Image via web URL
    logger.info("Image via web URL")
    stream = client.api_query(
        model=vision_model,
        user_msg="Should I take this fight?",
        img="https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest/scale-to-width-down/256?cb=20220523172438",
        stream=True,
    )
    print_stream(stream)

    # Image via local file path
    logger.info("Image via local file path")
    stream = client.api_query(
        model=vision_model, user_msg="Should I take this fight?", img=Path("./assets/example-img.webp"), stream=True
    )
    print_stream(stream)

    # Supports different file formats & does not require Path input
    logger.info("Image via local JPEG file path")
    stream = client.api_query(
        model=vision_model, user_msg="Analyze why this picture is funny.", img="./assets/example-img.jpeg", stream=True
    )
    print_stream(stream)

    # Supports multiple images per request (as list of Paths, bytes, NOT mixed)
    logger.info("Multiple images via local file paths")
    stream = client.api_query(
        model=vision_model,
        user_msg="Describe both images.",
        img=[Path("./assets/example-img.webp"), Path("./assets/example-img.jpeg")],
        stream=True,
    )
    print_stream(stream)

    # Supports raw bytes input
    logger.info("Image via raw bytes")
    with open("./assets/example-img.jpeg", "rb") as f:
        img_bytes = f.read()

    stream = client.api_query(model=vision_model, user_msg="What is in this image?", img=img_bytes, stream=True)
    print_stream(stream)

    """
    Advanced Configuration: System Prompts, JSON Mode & Parameters

        - any kwargs can be passed according to LiteLLM conventions
        - LiteLLM adheres to OpenAI API & translates as needed for other providers.
        - You can therefore use reasoning_effort for Gemini Models, although the Gemini docs require different syntax.
        - Provider- or model-specific parameters can also be passed via `extra_body` dict in kwargs.
    """
    logger.info("--- Advanced Configuration ---")
    logger.info("Advanced Configuration: JSON Output & Temperature")
    kwargs = {
        "max_tokens": 100,
        "reasoning_effort": "none",  # none | low | medium | high
    }
    response = client.api_query(
        model="openai/gpt-5.2",
        user_msg="List 3 primary colors in JSON format: {colors: []}",
        system_prompt="You are a JSON-only assistant.",
        response_format={"type": "json_object"},
        **kwargs,
    )
    response = client.api_query(
        model="gemini/gemini-2.5-flash-lite",
        user_msg="List 3 primary colors in JSON format: {colors: []}",
        system_prompt="You are a JSON-only assistant.",
        response_format={"type": "json_object"},
        **kwargs,
    )
    logger.info("JSON Response:\n" + response.choices[0].message.content + "\n")

    """
    Embeddings Generation

    Simple RAG Retrieval Example
        - for client-compatible RAG database refer to  https://github.com/patrickab/rag-database
    """
    logger.info("--- Embeddings Generation ---")
    logger.info("Generating Embeddings for RAG Retrieval")

    embedding_model = "ollama/embeddinggemma:300m"
    dummy_query = "Tell me something weird about pineapple pizza"
    dummy_docs = [
        "Pizza was invented in Naples, Italy in the 18th century.",
        "The most popular pizza topping in America is pepperoni.",
        "Pineapple on pizza is called Hawaiian pizza and was invented in Canada.",
        "The world's most expensive pizza costs $12,000 and includes gold flakes.",
    ]

    # Generate embeddings for knowledge base and query
    db_embeddings = client.get_embedding(model=embedding_model, input_text=dummy_docs)
    db_embeddings = np.array([emb["embedding"] for emb in db_embeddings["data"]])
    query_embedding = client.get_embedding(model=embedding_model, input_text=dummy_query)
    query_embedding = np.array(query_embedding["data"][0]["embedding"])

    # Simple cosine similarity to find most relevant document (normally you'd use a vector DB)
    similarities = [
        np.dot(query_embedding, np.array(emb)) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(emb)))
        for emb in db_embeddings
    ]
    best_match_idx = np.argmax(similarities)
    retrieved_context = dummy_docs[best_match_idx]

    logger.info(f"Query: '{dummy_query}'")
    logger.info(f"Retrieved Context: '{retrieved_context}'")

    """
    Multimodal Batch Processing

    Asynchronous batch processing
        - maximizes throughput
        - Does not reduce API costs for commercial providers
    """
    logger.info("--- Multimodal Batch Processing ---")
    logger.info("Running Batch Multimodal Analysis")

    img_nature = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    img_space = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Bruce_McCandless_II_during_EVA_in_1984.jpg/1024px-Bruce_McCandless_II_during_EVA_in_1984.jpg"

    # Construct a batch of diverse requests
    # The client handles provider-specific formatting automatically.
    # Request 1: Analyze a single image
    # Request 2: Compare two images (Text + Multiple Images)
    # Request 3: Pure visual captioning (No user text, just system prompt + image)
    batch_workload = [
        {"user_msg": "Describe the weather and atmosphere in this image.", "img": img_nature},
        {"user_msg": "Compare the environments in these two images. Which one is on Earth?", "img": [img_nature, img_space]},
        {"system_prompt": "You are a poetic caption generator. Output only a haiku.", "img": img_space},
    ]

    # Execute parallel batch
    # This runs concurrently, maximizing throughput on vLLM or avoiding blocking on Cloud APIs
    logger.info(f"Dispatching {len(batch_workload)} multimodal requests to {vision_model}...")

    batch_results = client.batch_api_query(requests=batch_workload, model=vision_model, temperature=0.2, max_tokens=100)

    # Process results
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Request {i} failed: {result}")
        else:
            logger.info(f"Result {i}: {result.choices[0].message.content.strip()}")

    """
    Optimizations for high-performance local inference with limited VRAM.

    Demonstrates how to use quantization
    & memory optimizations to...
    - ...use a 12GB VRAM GPU to load a 28GB model...
    - ...& simultaneously process 8 images in parallel

    - AWQ 4-bit: Compresses weights to ~9.5GB VRAM - 1% most important parameters are kept as 16-bit float, rest is 4-bit float.
    - Enforce Eager: Disables CUDA Graphs; saves ~500MB buffer at slight inference speed cost (-10% to -15%).
    - VRAM Utilization: Allocates max available VRAM, leaving only ~600MB for desktop/overhead.
    - Max Inference Tokens: Limits KV cache size to prevent OOM during profiling.
    """
    # --------------------- Prepare optimization parameters --------------------- #
    logger.info("Processing with custom vLLM command...")
    quantized_model = "cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Max VRAM utilization (0.95 leaves some overhead for system & other apps)
    gpu_mem_util = 0.95

    # Higher throughput (tokens per second) but more VRAM usage
    # Defines number of simultaneous forward passes
    max_parallel_requests = 8

    # Restrict max tokens to allow parallelism within VRAM limits
    # Default: vLLM uses maximum model context length & upon startup
    #          allocates theoretical maximum of max_tokens*max_requests
    # input + output tokens = max_tokens
    # large amount of tokens is needed for images
    max_tokens = 1024

    # approximate tokens for image input
    img_max_tokens = 512

    # --------------------- Prepare vLLM Server start command --------------------- #
    vllm_cmd = f"vllm serve {quantized_model} --port 8000 --gpu-memory-utilization {gpu_mem_util} --max-model-len {max_tokens} --max-num-seqs {max_parallel_requests} --enforce-eager"  # noqa
    vllm_cmd = vllm_cmd.split()

    # ----------------------- Optimize images for Ministral ----------------------- #

    PATCH_SIZE = 14  # Ministral/Pixtral patch size - refer to model card for details

    def estimate_ministral_tokens(width: int, height: int) -> tuple[int, tuple[int, int]]:
        """
        Estimates token usage for Ministral 3 / Pixtral models.

        Formula:
        1. Calculate number of 14x14 patches in each dimension.
        2. Total = (patches_w * patches_h) + patches_h (row break tokens)
        """
        import math

        # Ceiling division to account for partial patches (which get padded)
        patches_w = math.ceil(width / PATCH_SIZE)
        patches_h = math.ceil(height / PATCH_SIZE)

        total_tokens = (patches_w * patches_h) + patches_h
        return total_tokens, (patches_w, patches_h)

    def optimize_image_for_ministral(image_url: str, max_tokens: int) -> dict:
        """For interactive notebook refer to src/examples/"""
        import base64
        import io
        import math

        from PIL import Image
        import requests

        # 1. Download Image with User-Agent header to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"  # noqa
        }

        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        original_size = img.size

        # 2. Iteratively find the best scale to fit the token budget
        current_tokens, _ = estimate_ministral_tokens(img.width, img.height)

        if current_tokens > max_tokens:
            scale_factor = math.sqrt(max_tokens / current_tokens)

            # Calculate new raw dimensions
            new_w = int(img.width * scale_factor)
            new_h = int(img.height * scale_factor)

            # 3. Snap to Grid (Multiples of 14)
            new_w = round(new_w / PATCH_SIZE) * PATCH_SIZE
            new_h = round(new_h / PATCH_SIZE) * PATCH_SIZE

            # Ensure we don't shrink to 0
            new_w = max(new_w, PATCH_SIZE)
            new_h = max(new_h, PATCH_SIZE)

            # 4. Resize using LANCZOS
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 5. Convert to Base64
        buffered = io.BytesIO()
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        final_tokens, _ = estimate_ministral_tokens(img.width, img.height)

        return {
            "original_size": original_size,
            "original_tokens": current_tokens,
            "optimized_size": img.size,
            "optimized_tokens": final_tokens,
            "base64_url": f"data:image/jpeg;base64,{img_str}",
        }

    dict_img_nature_optimized = optimize_image_for_ministral(img_nature, max_tokens=img_max_tokens)
    dict_img_space_optimized = optimize_image_for_ministral(img_space, max_tokens=img_max_tokens)
    batch_workload = [
        {"user_msg": "Describe the weather and atmosphere in this image.", "img": dict_img_nature_optimized["base64_url"]},
        {
            "user_msg": "Compare the environments in these two images. Which one is on Earth?",
            "img": [dict_img_nature_optimized["base64_url"], dict_img_space_optimized["base64_url"]],
        },
        {"system_prompt": "You are a poetic caption generator. Output only a haiku.", "img": dict_img_space_optimized["base64_url"]},
    ]
    batch_results = client.batch_api_query(
        requests=batch_workload,
        model=f"hosted_vllm/{quantized_model}",  # litellm requires hosted_vllm/ prefix
        vllm_cmd=vllm_cmd,
        temperature=0.2,
        max_workers=max_parallel_requests,
    )
    logger.info("Processing complete. Results:")
    logger.info(
        f"--- Optimization Results ---\n"
        f"Original Size:   {dict_img_nature_optimized['original_size']} px\n"
        f"Original Cost:   ~{dict_img_nature_optimized['original_tokens']} tokens\n"
        f"{'-' * 30}\n"
        f"Target Cost:     ~{img_max_tokens} tokens\n"
        f"Optimized Size:  {dict_img_nature_optimized['optimized_size']} px (Multiple of 14? Yes)\n"
        f"Optimized Cost:  ~{dict_img_nature_optimized['optimized_tokens']} tokens"
    )
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Request {i} failed: {result}")
        else:
            logger.info(f"Assistant Response {i}: {result.choices[0].message.content.strip()}")

    # Cleanup explicitly
    client.kill_inference_engines()


if __name__ == "__main__":
    main()
