from pathlib import Path

import numpy as np

from llm_baseclient.client import LLMClient
from llm_baseclient.logger import get_logger

logger = get_logger()


def main() -> None:
    client = LLMClient()

    """Stateless API call (non-streaming)"""
    # ------------------- Commercial Providers ------------------- #
    # Assumes API keys set in environment variables using provider conventions.
    response = client.api_query(model="openai/gpt-5.2", user_msg="Hello, world!")
    response = client.api_query(model="gemini/gemini-2.5-flash", user_msg="Hello, world!")

    # --------------- Open-Source / Local Providers -------------- #
    # Assumes vLLM is installed - automatically downloads model if not present.
    response = client.api_query(model="hosted_vllm/Qwen/Qwen3-0.6B", user_msg="Hello, world!")
    # Assumes Ollama is installed - automatically downloads model if not present.
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
    response = client.chat(model="ollama/gemma3:4b", user_msg="Tell me a knock knock joke.", stream=False)
    response = client.chat(model="ollama/gemma3:4b", user_msg="Who's there?", stream=False)
    logger.info("Message History:\n\n" + str(client.messages))

    """Image input examples"""
    # Adjust model as needed - Ministral-3 runs fast on weak laptop CPUs
    visual_model = "ollama/ministral-3:3b"

    def print_stream(stream) -> None:  # noqa
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

    """Advanced Configuration: System Prompts, JSON Mode & Parameters"""
    logger.info("Advanced Configuration: JSON Output & Temperature")
    # Any llm specific kwargs can be passed according to LiteLLM conventions.
    # LiteLLM adheres to OpenAI API & translates as needed for other providers.
    # Using Litellm you can therefore use reasoning_effort for Gemini Models, although the Gemini docs require different syntax.
    # Provider- or model-specific parameters can also be passed via `extra_body` dict in kwargs.
    kwargs = {
        "max_tokens": 100,
        "reasoning_effort": "none" # none | low | medium | high
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

    """Embeddings Generation - RAG Retrieval Example"""
    logger.info("Generating Embeddings for RAG Retrieval")
    
    # Simulate a tiny knowledge base about pizza
    embedding_model = "ollama/embeddinggemma:300m"
    dummy_docs = [
        "Pizza was invented in Naples, Italy in the 18th century.",
        "The most popular pizza topping in America is pepperoni.",
        "Pineapple on pizza is called Hawaiian pizza and was invented in Canada.",
        "The world's most expensive pizza costs $12,000 and includes gold flakes.",
    ]
    
    dummy_query = "Tell me something weird about pineapple pizza"

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

    # Cleanup explicitly
    client.kill_inference_engines()


if __name__ == "__main__":
    main()
