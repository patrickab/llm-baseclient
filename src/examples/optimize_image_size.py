# ruff: noqa: E501, ANN201
# %% [markdown]
# # Ministral Image Token Optimization
#
# This notebook demonstrates how to programmatically resize images to fit a specific token budget
# for the **Ministral 3** (and Pixtral) family of models.
#
# ### Why optimize?
# 1. **VRAM Limits**: High-res images (e.g., 1024x1024) consume ~5000+ tokens. This can OOM (Out Of Memory) vLLM if the context window is small (e.g., 2048 or 4096).
# 2. **Cost/Latency**: Fewer tokens = faster inference and lower API costs.
# 3. **Padding Efficiency**: Ministral processes images in $14 \times 14$ patches. Dimensions that are not multiples of 14 result in wasted padding.

# %%
import base64
import io
import math

from PIL import Image
import requests

# %% [markdown]
# ### 1. Configuration
# *   **PATCH_SIZE**: Ministral uses 14x14 patches.
# *   **TARGET_TOKENS**: The approximate budget we want to hit (e.g., ~1000 tokens).

# %%
PATCH_SIZE = 14  # Ministral/Pixtral patch size
TARGET_TOKENS = 1200  # Desired max token usage per image

# %% [markdown]
# ### 2. Token Estimation Logic
# Ministral's vision encoder (Pixtral-based) works differently than LLaVA.
# *   It preserves aspect ratio (no forced squares).
# *   It adds a "break" token at the end of every row of patches.
#
# $$ \text{Tokens} \approx (\text{Rows} \times \text{Cols}) + \text{Rows} $$


# %%
def estimate_ministral_tokens(width: int, height: int) -> tuple[int, tuple[int, int]]:
    """
    Estimates token usage for Ministral 3 / Pixtral models.

    Formula:
    1. Calculate number of 14x14 patches in each dimension.
    2. Total = (patches_w * patches_h) + patches_h (row break tokens)
    """
    # Ceiling division to account for partial patches (which get padded)
    patches_w = math.ceil(width / PATCH_SIZE)
    patches_h = math.ceil(height / PATCH_SIZE)

    total_tokens = (patches_w * patches_h) + patches_h
    return total_tokens, (patches_w, patches_h)


# %% [markdown]
# ### 3. Optimization Algorithm
# This function performs the resizing with three specific constraints:
# 1.  **Aspect Ratio Preservation**: Never crop; only scale.
# 2.  **Grid Snapping**: Round dimensions to the nearest multiple of 14 to eliminate padding waste.
# 3.  **Lanczos Resampling**: Use a high-quality filter to preserve "signal" (edges/text) even at lower resolutions.


# %%
def optimize_image_for_ministral(image_url: str, max_tokens: int = TARGET_TOKENS) -> dict:
    # 1. Download Image with User-Agent header to avoid 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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


# %% [markdown]
# ### 4. Execution & Verification
# We test with a high-resolution image (Wikimedia Commons) to see the reduction.

# %%
# Example: A high-res image of nature (often 4000px+)
TEST_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
MAX_TOKENS = 500

result = optimize_image_for_ministral(TEST_URL, max_tokens=MAX_TOKENS)

print("--- Optimization Results ---")
print(f"Original Size:   {result['original_size']} px")
print(f"Original Cost:   ~{result['original_tokens']} tokens")
print("-" * 30)
print(f"Target Cost:     ~{MAX_TOKENS} tokens")
print(f"Optimized Size:  {result['optimized_size']} px (Multiple of 14? Yes)")
print(f"Optimized Cost:  ~{result['optimized_tokens']} tokens")
print(f"Reduction:       {100 - (result['optimized_tokens'] / result['original_tokens'] * 100):.1f}% fewer tokens")

# %%
# Display original image
#headers = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#}
#original_img = Image.open(io.BytesIO(requests.get(TEST_URL, headers=headers).content))
#original_img.show()

# %%
# Display optimized image
# img_data = base64.b64decode(result["base64_url"].split(",")[1])
# img = Image.open(io.BytesIO(img_data))
# img.show()

# %% [markdown]
# ### 5. Usage in LiteLLM
# You can now pass `result['base64_url']` directly to your batch workload.

# %%
# Example payload construction
payload = {
    "role": "user",
    "content": [{"type": "text", "text": "Describe this image."}, {"type": "image_url", "image_url": {"url": result["base64_url"]}}],
}
print("Payload ready for LiteLLM.")
