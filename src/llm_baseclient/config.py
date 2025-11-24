from typing import List

# Anthropic Models (Updated for Nov 2025)
# Note: Claude 3.5 Sonnet (20241022) is officially retired/legacy in this timeline.
MODELS_ANTHROPIC: List[str] = [
    # Flagship / High Intelligence
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    
    # High Speed / Cost Efficient
    "claude-haiku-4-5-20251015",
    "claude-haiku-4-5",  # Auto-alias
    
    # Legacy / Previous Generation (Maintenance Mode)
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307"
]

# OpenAI Models (Updated for Nov 2025)
MODELS_OPENAI: List[str] = [
    # Reasoning & Research (o-series)
    "o3-deep-research",
    "o1",             # Stable release
    "o1-mini",
    "o4-mini",        # Fast reasoning
    
    # Flagship (GPT-5 / GPT-4 series)
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-4.5-preview",
    "gpt-4o",
    "chatgpt-4o-latest",
    
    # Legacy / Utility
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "text-embedding-3-large",
    "text-embedding-3-small"
]

# Google Gemini Models (Updated for Nov 2025)
MODELS_GEMINI: List[str] = [
    # Gemini 3.0 & 2.5 (Current Flagships)
    "gemini-3.0-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash-thinking-exp",
    
    # Gemini 1.5 (Stable Legacy)
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    
    # Embeddings & Specialized
    "text-embedding-001",
    "aqa"
]
