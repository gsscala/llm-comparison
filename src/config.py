"""Central model registry and prompt configuration for the 24,564 MiB VRAM sentiment benchmark."""

VRAM_BUDGET_MIB = 24564  # 24,564 MiB binary == 25.76 GB decimal (Ollama unit)

PROMPT_JSON_SST5 = """Classify the sentiment of the following movie review into one of five ordinal categories:
0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive

Return a JSON object with a single integer field "sentiment" in {0, 1, 2, 3, 4}.

Review: "{sentence}"
"""

PROMPT_JSON_SEMEVAL = """Classify the sentiment of the tweet towards the given topic into one of five ordinal categories:
0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive

Return a JSON object with a single integer field "sentiment" in {0, 1, 2, 3, 4}.

{sentence}
"""

PROMPT_THINK_SST5 = """Classify the sentiment of the following movie review into one of five ordinal categories:
0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive

After your reasoning, output ONLY a JSON object of the form {{"sentiment": <integer 0-4>}}.

Review: "{sentence}"
"""

PROMPT_THINK_SEMEVAL = """Classify the sentiment of the tweet towards the given topic into one of five ordinal categories:
0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive

After your reasoning, output ONLY a JSON object of the form {{"sentiment": <integer 0-4>}}.

{sentence}
"""

# 30 curated frontier open-weight models across 11 organizations for a 24,564 MiB GPU
MAIN_LLM_MODELS = [
    # Google Gemma 4 (Dense + MoE scaling curve)
    {
        "tag": "gemma4:e2b",
        "family": "Gemma 4",
        "lab": "Google",
        "params_b": 2.0,
        "active_params_b": 2.0,
        "arch": "Dense",
        "ollama_size_gb": 4.6,
        "est_mib": 4387,
        "supports_think": True,
        "role": "Sub-3B edge scaling",
    },
    {
        "tag": "gemma4:e4b",
        "family": "Gemma 4",
        "lab": "Google",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "arch": "Dense",
        "ollama_size_gb": 6.6,
        "est_mib": 6294,
        "supports_think": True,
        "role": "4B scaling tier",
    },
    {
        "tag": "gemma4:12b",
        "family": "Gemma 4",
        "lab": "Google",
        "params_b": 12.0,
        "active_params_b": 12.0,
        "arch": "Dense",
        "ollama_size_gb": 7.7,
        "est_mib": 7343,
        "supports_think": True,
        "role": "12B mid-size tier",
    },
    {
        "tag": "gemma4:26b",
        "family": "Gemma 4",
        "lab": "Google",
        "params_b": 26.0,
        "active_params_b": 3.8,
        "arch": "MoE",
        "ollama_size_gb": 16.0,
        "est_mib": 15259,
        "supports_think": True,
        "role": "26B-A3.8B MoE vs. Dense comparison",
    },
    {
        "tag": "gemma4:31b",
        "family": "Gemma 4",
        "lab": "Google",
        "params_b": 31.0,
        "active_params_b": 31.0,
        "arch": "Dense",
        "ollama_size_gb": 19.0,
        "est_mib": 18120,
        "supports_think": True,
        "role": "31B dense single-GPU ceiling",
    },
    # Alibaba Qwen 3.5 / 3.6 / 3.8 (Dense + MoE scaling curve)
    {
        "tag": "qwen3.5:2b",
        "family": "Qwen 3.5/3.8",
        "lab": "Alibaba",
        "params_b": 2.0,
        "active_params_b": 2.0,
        "arch": "Dense",
        "ollama_size_gb": 2.7,
        "est_mib": 2575,
        "supports_think": True,
        "role": "Sub-3B edge scaling",
    },
    {
        "tag": "qwen3.5:4b",
        "family": "Qwen 3.5/3.8",
        "lab": "Alibaba",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "arch": "Dense",
        "ollama_size_gb": 3.3,
        "est_mib": 3147,
        "supports_think": True,
        "role": "4B scaling tier",
    },
    {
        "tag": "qwen3.5:9b",
        "family": "Qwen 3.5/3.8",
        "lab": "Alibaba",
        "params_b": 9.0,
        "active_params_b": 9.0,
        "arch": "Dense",
        "ollama_size_gb": 6.6,
        "est_mib": 6294,
        "supports_think": True,
        "role": "9B mid-size tier",
    },
    {
        "tag": "qwen3.8:27b",
        "family": "Qwen 3.5/3.8",
        "lab": "Alibaba",
        "params_b": 27.0,
        "active_params_b": 27.0,
        "arch": "Dense",
        "ollama_size_gb": 18.0,
        "est_mib": 17166,
        "supports_think": True,
        "role": "27B dense flagship",
    },
    {
        "tag": "qwen3.6:35b",
        "family": "Qwen 3.5/3.8",
        "lab": "Alibaba",
        "params_b": 35.0,
        "active_params_b": 3.0,
        "arch": "MoE",
        "ollama_size_gb": 23.0,
        "est_mib": 21934,
        "supports_think": True,
        "role": "35B-A3B MoE ceiling",
    },
    # Meta Llama 3 & Muse Glimmer
    {
        "tag": "llama3.2:3b",
        "family": "Llama / Muse",
        "lab": "Meta",
        "params_b": 3.0,
        "active_params_b": 3.0,
        "arch": "Dense",
        "ollama_size_gb": 2.0,
        "est_mib": 1907,
        "supports_think": False,
        "role": "Reference 2024 3B baseline",
    },
    {
        "tag": "llama3.1:8b",
        "family": "Llama / Muse",
        "lab": "Meta",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "arch": "Dense",
        "ollama_size_gb": 4.9,
        "est_mib": 4673,
        "supports_think": False,
        "role": "Reference 2024 8B baseline",
    },
    {
        "tag": "muse-glimmer:30b",
        "family": "Llama / Muse",
        "lab": "Meta",
        "params_b": 30.0,
        "active_params_b": 30.0,
        "arch": "Dense",
        "ollama_size_gb": 18.0,
        "est_mib": 17166,
        "supports_think": True,
        "role": "Meta Aug 2026 30B single-GPU flagship",
    },
    # OpenAI Open-Weight
    {
        "tag": "gpt-oss:20b",
        "family": "gpt-oss",
        "lab": "OpenAI",
        "params_b": 21.0,
        "active_params_b": 3.6,
        "arch": "MoE",
        "ollama_size_gb": 14.0,
        "est_mib": 13351,
        "supports_think": True,
        "role": "OpenAI 20B-A3.6B open-weight MoE",
    },
    # Mistral / Ministral 3
    {
        "tag": "ministral-3:3b",
        "family": "Mistral 3",
        "lab": "Mistral",
        "params_b": 3.0,
        "active_params_b": 3.0,
        "arch": "Dense",
        "ollama_size_gb": 3.0,
        "est_mib": 2861,
        "supports_think": False,
        "role": "3B edge tier",
    },
    {
        "tag": "ministral-3:8b",
        "family": "Mistral 3",
        "lab": "Mistral",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "arch": "Dense",
        "ollama_size_gb": 6.0,
        "est_mib": 5722,
        "supports_think": False,
        "role": "8B tier",
    },
    {
        "tag": "ministral-3:14b",
        "family": "Mistral 3",
        "lab": "Mistral",
        "params_b": 14.0,
        "active_params_b": 14.0,
        "arch": "Dense",
        "ollama_size_gb": 9.1,
        "est_mib": 8678,
        "supports_think": False,
        "role": "14B mid-size tier",
    },
    {
        "tag": "mistral-small3.2:24b",
        "family": "Mistral 3",
        "lab": "Mistral",
        "params_b": 24.0,
        "active_params_b": 24.0,
        "arch": "Dense",
        "ollama_size_gb": 15.0,
        "est_mib": 14305,
        "supports_think": False,
        "role": "24B single-GPU Mistral ceiling",
    },
    # IBM Granite 4.2 (Mamba-2 / Transformer Hybrid)
    {
        "tag": "granite4.2:3b",
        "family": "Granite 4.2",
        "lab": "IBM",
        "params_b": 3.0,
        "active_params_b": 3.0,
        "arch": "Hybrid",
        "ollama_size_gb": 2.2,
        "est_mib": 2098,
        "supports_think": True,
        "role": "Enterprise hybrid 3B tier",
    },
    {
        "tag": "granite4.2:8b",
        "family": "Granite 4.2",
        "lab": "IBM",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "arch": "Hybrid",
        "ollama_size_gb": 5.3,
        "est_mib": 5054,
        "supports_think": True,
        "role": "Enterprise hybrid 8B tier",
    },
    {
        "tag": "granite4.2:30b",
        "family": "Granite 4.2",
        "lab": "IBM",
        "params_b": 30.0,
        "active_params_b": 30.0,
        "arch": "Hybrid",
        "ollama_size_gb": 18.0,
        "est_mib": 17166,
        "supports_think": True,
        "role": "Enterprise hybrid 30B tier",
    },
    # NVIDIA Nemotron 3 / 3.5 (Mamba-Attention Hybrid & MoE)
    {
        "tag": "nemotron-3-nano:4b",
        "family": "Nemotron 3",
        "lab": "NVIDIA",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "arch": "Hybrid",
        "ollama_size_gb": 2.8,
        "est_mib": 2670,
        "supports_think": True,
        "role": "Compact 4B hybrid model",
    },
    {
        "tag": "nemotron-3.5-lightning:30b",
        "family": "Nemotron 3",
        "lab": "NVIDIA",
        "params_b": 30.0,
        "active_params_b": 3.0,
        "arch": "Hybrid MoE",
        "ollama_size_gb": 25.0,
        "est_mib": 23842,
        "supports_think": True,
        "role": "30B-A3B hybrid Mamba-Attention MoE",
    },
    # Liquid AI LFM 2.5 (Conv + Attention Hybrid)
    {
        "tag": "lfm2.5:8b",
        "family": "LFM 2.5",
        "lab": "Liquid AI",
        "params_b": 8.0,
        "active_params_b": 1.0,
        "arch": "Hybrid MoE",
        "ollama_size_gb": 5.2,
        "est_mib": 4959,
        "supports_think": False,
        "role": "8B-A1B ultra-fast edge MoE",
    },
    {
        "tag": "lfm2.5:24b",
        "family": "LFM 2.5",
        "lab": "Liquid AI",
        "params_b": 24.0,
        "active_params_b": 24.0,
        "arch": "Hybrid",
        "ollama_size_gb": 15.0,
        "est_mib": 14305,
        "supports_think": False,
        "role": "24B non-transformer hybrid",
    },

    # Z.ai GLM
    {
        "tag": "glm-4.7-flash",
        "family": "GLM 4",
        "lab": "Z.ai",
        "params_b": 30.0,
        "active_params_b": 3.0,
        "arch": "MoE",
        "ollama_size_gb": 19.0,
        "est_mib": 18120,
        "supports_think": True,
        "role": "30B-class Flash MoE",
    },
    # Microsoft Phi-4
    {
        "tag": "phi4-mini:3.8b",
        "family": "Phi 4",
        "lab": "Microsoft",
        "params_b": 3.8,
        "active_params_b": 3.8,
        "arch": "Dense",
        "ollama_size_gb": 2.5,
        "est_mib": 2384,
        "supports_think": False,
        "role": "Synthetic-data 3.8B tier",
    },
    {
        "tag": "phi4:14b",
        "family": "Phi 4",
        "lab": "Microsoft",
        "params_b": 14.0,
        "active_params_b": 14.0,
        "arch": "Dense",
        "ollama_size_gb": 9.1,
        "est_mib": 8678,
        "supports_think": False,
        "role": "Synthetic-data 14B tier",
    },
    # Cohere Aya Expanse
    {
        "tag": "aya-expanse:8b",
        "family": "Aya Expanse",
        "lab": "Cohere",
        "params_b": 8.0,
        "active_params_b": 8.0,
        "arch": "Dense",
        "ollama_size_gb": 5.1,
        "est_mib": 4864,
        "supports_think": False,
        "role": "Multilingual 8B tier",
    },
    {
        "tag": "aya-expanse:32b",
        "family": "Aya Expanse",
        "lab": "Cohere",
        "params_b": 32.0,
        "active_params_b": 32.0,
        "arch": "Dense",
        "ollama_size_gb": 20.0,
        "est_mib": 19073,
        "supports_think": False,
        "role": "Multilingual 32B ceiling",
    },
]

# Models evaluated with think=True (num_ctx=4096) for the Thinking vs. Non-Thinking Ablation
THINKING_ABLATION_MODELS = [
    {"tag": "gemma4:12b", "family": "Gemma 4", "params_b": 12.0, "arch": "Dense", "mode": "hybrid"},
    {"tag": "gemma4:26b", "family": "Gemma 4", "params_b": 26.0, "arch": "MoE", "mode": "hybrid"},
    {"tag": "qwen3.5:9b", "family": "Qwen 3.5/3.8", "params_b": 9.0, "arch": "Dense", "mode": "hybrid"},
    {"tag": "qwen3.8:27b", "family": "Qwen 3.5/3.8", "params_b": 27.0, "arch": "Dense", "mode": "hybrid"},
    {"tag": "muse-glimmer:30b", "family": "Llama / Muse", "params_b": 30.0, "arch": "Dense", "mode": "hybrid"},
    {"tag": "deepseek-r1:14b", "family": "DeepSeek R1", "params_b": 14.0, "arch": "Dense", "mode": "reasoning_only"},
    {"tag": "deepseek-r1:32b", "family": "DeepSeek R1", "params_b": 32.0, "arch": "Dense", "mode": "reasoning_only"},
    {"tag": "phi4-reasoning:14b", "family": "Phi 4", "params_b": 14.0, "arch": "Dense", "mode": "reasoning_only"},
]

# Zero-shot encoder baselines (Cross-Encoder NLI and Single-Pass GLiClass)
ENCODER_MODELS = [
    {
        "hf_id": "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        "short_name": "ModernBERT-base-zeroshot-v2.0",
        "family": "Encoder (NLI)",
        "arch": "Encoder",
        "encoder_type": "nli",
        "params_b": 0.149,
    },
    {
        "hf_id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        "short_name": "ModernBERT-large-zeroshot-v2.0",
        "family": "Encoder (NLI)",
        "arch": "Encoder",
        "encoder_type": "nli",
        "params_b": 0.395,
    },
    {
        "hf_id": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "short_name": "deberta-v3-large-zeroshot-v2.0",
        "family": "Encoder (NLI)",
        "arch": "Encoder",
        "encoder_type": "nli",
        "params_b": 0.435,
    },
    {
        "hf_id": "MoritzLaurer/bge-m3-zeroshot-v2.0",
        "short_name": "bge-m3-zeroshot-v2.0",
        "family": "Encoder (NLI)",
        "arch": "Encoder",
        "encoder_type": "nli",
        "params_b": 0.568,
    },
    {
        "hf_id": "knowledgator/gliclass-base-v1.0",
        "short_name": "gliclass-base-v1.0",
        "family": "Encoder (GLiClass)",
        "arch": "Encoder",
        "encoder_type": "gliclass",
        "params_b": 0.184,
    },
    {
        "hf_id": "knowledgator/gliclass-large-v1.0",
        "short_name": "gliclass-large-v1.0",
        "family": "Encoder (GLiClass)",
        "arch": "Encoder",
        "encoder_type": "gliclass",
        "params_b": 0.435,
    },
]


def get_model_metadata(model_identifier: str) -> dict:
    """Resolve family, architecture, and parameter metadata for both new and legacy model identifiers."""
    clean_id = model_identifier.replace("_sst", "").replace("_semeval", "").replace("_think", "")
    colon_id = clean_id.replace("_", ":", 1) if "_" in clean_id and ":" not in clean_id else clean_id

    for entry in MAIN_LLM_MODELS + THINKING_ABLATION_MODELS:
        if entry["tag"] in (clean_id, colon_id) or entry["tag"].replace(":", "_") == clean_id:
            return {
                "family": entry["family"],
                "arch": entry["arch"],
                "params_b": entry["params_b"],
                "active_params_b": entry.get("active_params_b", entry["params_b"]),
            }

    for enc in ENCODER_MODELS:
        if enc["short_name"] in clean_id or enc["hf_id"] in clean_id:
            return {
                "family": enc["family"],
                "arch": "Encoder",
                "params_b": enc["params_b"],
                "active_params_b": enc["params_b"],
            }

    # Legacy fallback rules for historical runs in SUMMARIZED_DATA
    lower = clean_id.lower()
    if any(k in lower for k in ["deberta", "roberta", "bart", "minilm", "albert", "bge", "modernbert", "gliclass"]):
        return {"family": "Encoder (NLI)", "arch": "Encoder", "params_b": 0.35, "active_params_b": 0.35}
    if "gemma" in lower:
        return {"family": "Gemma (Legacy)" if "gemma4" not in lower else "Gemma 4", "arch": "Dense", "params_b": 9.0, "active_params_b": 9.0}
    if "qwen" in lower:
        is_moe = "30b" in lower or "35b" in lower
        return {
            "family": "Qwen (Legacy)" if not any(x in lower for x in ["3.5", "3.6", "3.8"]) else "Qwen 3.5/3.8",
            "arch": "MoE" if is_moe else "Dense",
            "params_b": 14.0,
            "active_params_b": 3.0 if is_moe else 14.0,
        }
    if "llama" in lower or "muse" in lower or "vicuna" in lower:
        return {"family": "Llama / Muse", "arch": "Dense", "params_b": 8.0, "active_params_b": 8.0}
    if "mistral" in lower or "ministral" in lower or "zephyr" in lower:
        return {"family": "Mistral 3", "arch": "Dense", "params_b": 14.0, "active_params_b": 14.0}
    if "granite" in lower:
        return {"family": "Granite 4.2", "arch": "Hybrid", "params_b": 8.0, "active_params_b": 8.0}
    if "nemotron" in lower:
        return {"family": "Nemotron 3", "arch": "Hybrid", "params_b": 4.0, "active_params_b": 4.0}
    if "lfm" in lower:
        return {"family": "LFM 2.5", "arch": "Hybrid", "params_b": 8.0, "active_params_b": 8.0}
    if "olmo" in lower:
        return {"family": "OLMo 3", "arch": "Dense", "params_b": 13.0, "active_params_b": 13.0}
    if "phi" in lower:
        return {"family": "Phi 4", "arch": "Dense", "params_b": 14.0, "active_params_b": 14.0}
    if "deepseek" in lower:
        return {"family": "DeepSeek R1", "arch": "Dense", "params_b": 14.0, "active_params_b": 14.0}
    if "aya" in lower:
        return {"family": "Aya Expanse", "arch": "Dense", "params_b": 8.0, "active_params_b": 8.0}
    if "glm" in lower:
        return {"family": "GLM 4", "arch": "MoE", "params_b": 30.0, "active_params_b": 3.0}
    if "gpt-oss" in lower:
        return {"family": "gpt-oss", "arch": "MoE", "params_b": 21.0, "active_params_b": 3.6}

    return {"family": "Other", "arch": "Dense", "params_b": 7.0, "active_params_b": 7.0}
