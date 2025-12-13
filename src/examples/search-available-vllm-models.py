# %% [markdown]
# # 🔍 vLLM Model + Architecture Explorer
#
# **Prerequisites:**
# - `pip install vllm huggingface_hub pandas`
# - (Optional) `huggingface-cli login` for gated models

# %%
import os

from huggingface_hub import HfApi, scan_cache_dir
import pandas as pd
import vllm
from vllm.model_executor.models import ModelRegistry

# Initialize API
api = HfApi()

print(f"vLLM Version: {vllm.__version__}")

# %% [markdown]
# # 1️⃣ Authentication Check
# Checks environment variables and API login status.

# %%
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
print(f"HF Token detected in ENV: {'✅ Yes' if token else '❌ No'}")

try:
    who = api.whoami()
    print(f"Authenticated as: {who.get('name')} ({who.get('type')})")
except Exception:
    print("⚠️ Not authenticated. You will only see public models.")

# %% [markdown]
# # 2️⃣ Retrieve vLLM Supported Architectures
# **Fix:** Uses `ModelRegistry` instead of the deprecated `supported_models`.

# %%
# Get all registered models in vLLM
# inspect_model_cls_registry returns a dict {ArchName: ModelClass}
try:
    # Method for vLLM >= 0.5.0
    supported_archs = list(ModelRegistry.get_supported_archs())
except AttributeError:
    # Fallback for older versions
    print("⚠️ Warning: Using older vLLM architecture detection.")
    from vllm.model_executor.model_loader import _MODEL_REGISTRY
    supported_archs = list(_MODEL_REGISTRY.keys())

print(f"✅ Found {len(supported_archs)} supported architectures.")

# Print examples in batches of 5
for i in range(0, len(supported_archs), 5):
    batch = supported_archs[i:i+5]
    print(f"Architectures [{i}-{min(i+4, len(supported_archs)-1)}]: {batch}")

# %% [markdown]
# # 3️⃣ Fetch Popular Models from Hugging Face
# %%
print("⏳ Fetching top 2,000 trending models from HF (this takes a moment)...")

# We fetch config to see the architecture
models = api.list_models(
    library="transformers",
    sort="downloads",
    direction="-1",
    limit=1000,
    fetch_config=True
)

vllm_compatible = []

for m in models:
    # Safety check if config exists
    if not m.config:
        continue
    
    # Get model architecture list (e.g., ['LlamaForCausalLM'])
    model_archs = m.config.get("architectures", [])
    
    # Check if ANY of the model's architectures are in vLLM's supported list
    if any(arch in supported_archs for arch in model_archs):
        vllm_compatible.append({
            "Model ID": m.modelId,
            "Architecture": model_archs[0] if model_archs else "Unknown",
            "Downloads": m.downloads,
            "Gated": m.gated,
            "Private": m.private
        })

print(f"✅ Found {len(vllm_compatible)} vLLM-compatible models among the top 2,000.")

# %% [markdown]
# # 4️⃣ Browse Compatible Models (Dataframe)
# Uses Pandas for an interactive table.

# %%
df = pd.DataFrame(vllm_compatible)
# Show top 10 most downloaded compatible models
df.head(20)

# %% [markdown]
# # 5️⃣ Filter Tools
# Easy filtering for specific families or types.

# %%
def search_models(keyword: str) -> pd.DataFrame:
    subset = df[df["Model ID"].str.contains(keyword, case=False)]
    return subset[["Model ID", "Architecture", "Downloads"]]

print(f"Llama 3 Models: {len(search_models('Llama-3'))}")
print(f"Qwen Models:    {len(search_models('Qwen'))}")
print(f"Mistral Models: {len(search_models('Mistral'))}")
print(f"AWQ Quantized:  {len(search_models('awq'))}")
print(f"GPTQ Quantized: {len(search_models('gptq'))}")

# Example: Show top 5 AWQ models
print("\n--- Top AWQ Models ---")
print(search_models('awq').head(5))

# %% [markdown]
# # 6️⃣ Correct Local Cache Check
# **Fix:** Uses `scan_cache_dir` for a robust look at what is actually on disk.

# %%
print("🔍 Scanning local Hugging Face cache...")

try:
    hf_cache_info = scan_cache_dir()
    local_repos = []

    for repo in hf_cache_info.repos:
        # We only care about models, not datasets
        if repo.repo_type == "model":
            # Check for the primary revision (usually main)
            size_mb = repo.size_on_disk / (1024 * 1024)
            local_repos.append({
                "Model ID": repo.repo_id,
                "Size (MB)": f"{size_mb:.2f}",
                "Refs": [r.ref_name for r in repo.refs]
            })

    df_local = pd.DataFrame(local_repos)
    
    if not df_local.empty:
        # Check compatibility of local models against vLLM list
        # Note: This is a string match. For 100% accuracy we'd load config, 
        # but checking if the ID exists in our 'vllm_compatible' list is a good proxy.
        
        print(f"✅ Found {len(df_local)} cached models.")
        print(df_local.head(10))
    else:
        print("❌ No models found in default Hugging Face cache.")

except Exception as e:
    print(f"Error scanning cache: {e}")

# %%