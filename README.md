# LLM comparison: frontier open-weight models vs. zero-shot encoders

This repository benchmarks 32 frontier open-weight language models, 9 thinking ablation checkpoints, and 6 zero-shot transformer encoders on 5-class ordinal sentiment analysis under a single-GPU budget of 24,564 MiB VRAM.

Evaluations run on two datasets:
1. Stanford Sentiment Treebank (SST-5): 2,210 movie reviews with fine-grained ordinal ratings from 0 (very negative) to 4 (very positive).
2. SemEval-2017 Task 4B: 12,379 topic-oriented tweets with context hashtags, mapped to the same 0 to 4 scale.

---

## Repository architecture

```
llm-comparison/
├── .gitignore
├── README.md
├── data/
│   └── semeval.txt                      # SemEval-2017 Task 4B dataset
├── notebooks/
│   └── benchmark_analysis.ipynb         # Dedicated notebook for visualizations and tables
├── results/
│   ├── raw/                             # Per-instance JSON outputs for all evaluated models
│   ├── summarized/                      # Consolidated metric summaries for each run
│   └── encoders/                        # Raw predictions from zero-shot encoders
├── src/
│   ├── __init__.py                      # Package entry point
│   ├── config.py                        # Model registry, VRAM budgets, prompt templates
│   ├── predictor.py                     # Ollama API client with VRAM logging and parsing
│   ├── labeler.py                       # Dedicated runner for long-running labeling workloads
│   ├── metrics.py                       # Unbiased full-N ordinal and classification metrics
│   ├── summarize.py                     # Aggregation script converting raw outputs to summaries
│   ├── encoders.py                      # Transformer zero-shot evaluation pipeline
│   └── plotting.py                      # Plotting utilities and Pareto curve routines
└── tests/
    └── test_benchmark_pipeline.py       # Unit test suite verifying parsing, metrics, and configs
```

---

## Evaluation methodology

### 1. Full sample evaluation without survivorship bias
Many LLM benchmarks discard unparsable outputs or instances where models hallucinated invalid options. This practice distorts results in favor of brittle models. In this pipeline, every sample is evaluated across the full dataset:
- Invalid, malformed, or ambiguous responses receive a worst-case ordinal error penalty: `max(y_true, 4 - y_true)`.
- For Macro-F1, failed predictions are assigned to an out-of-distribution error class to penalize class recall.
- Quadratic Weighted Kappa maps invalid outputs to the opposite polarity to penalize inter-rater agreement.

### 2. Structured output enforcement
Inference uses Ollama's structured output schema to enforce a strict JSON schema:
```json
{
  "type": "object",
  "properties": {
    "sentiment": {"type": "integer", "enum": [0, 1, 2, 3, 4]}
  },
  "required": ["sentiment"],
  "additionalProperties": false
}
```

### 3. Thinking trace separation
For reasoning models (such as DeepSeek R1 and Phi-4 reasoning), thoughts generated within `<think>...</think>` tags are extracted and separated from the final classification label, preventing internal deliberation digits from corrupting output extraction.

### 4. GPU memory verification and eviction
To prevent silent CPU offloading, [src/predictor.py](file:///home/gsscala/code/llm-comparison/src/predictor.py) checks Ollama's active process table after a warmup pass. It logs the exact VRAM allocation and GPU residency percentage. Models are explicitly unloaded with a zero keep-alive command before subsequent weights are initialized.

---

## Supported models

The model catalog in [src/config.py](file:///home/gsscala/code/llm-comparison/src/config.py) covers architectures sized to fit inside a 24,564 MiB VRAM budget:
- Google Gemma 4: Dense checkpoints (2B, 4B, 12B, 31B) and Mixture of Experts (26B).
- Alibaba Qwen 3.5 / 3.8: Dense models (2B, 4B, 9B, 27B) and MoE architectures (35B).
- Meta Llama 3 & Muse: Llama 3.2 3B, Llama 3.1 8B, and Muse Glimmer 30B.
- OpenAI: `gpt-oss:20b` (MoE).
- Mistral: Ministral (3B, 8B, 14B) and Mistral Small 3.2 (24B).
- IBM Granite 4.2: Hybrid Mamba/Transformer checkpoints (3B, 8B, 30B).
- NVIDIA Nemotron 3 / 3.5: Hybrid dense and MoE models (4B, 30B).
- Liquid AI LFM 2.5: Liquid neural network hybrids (8B MoE, 24B).
- AI2 OLMo 3 / 3.1: Fully open data models (7B, 32B).
- Microsoft Phi-4: Phi-4 mini (3.8B), Phi-4 (14B), and Phi-4 reasoning (14B).
- Cohere Aya Expanse: Multilingual models (8B, 32B).
- Zero-Shot Encoders: ModernBERT (base/large), DeBERTa-v3 large, BGE-M3, and GLiClass.

---

## Getting started

### Installation
Clone the repository and install dependencies in a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or install torch, transformers, datasets, ollama, pandas, matplotlib, adjustText
```

### Running unit tests
Verify pipeline functionality, metric logic, and parsing routines:
```bash
python3 -m unittest discover tests
```

### Running benchmark labeling
Run model inference via the dedicated command-line runner [src/labeler.py](file:///home/gsscala/code/llm-comparison/src/labeler.py):
```bash
# Evaluate all 32 frontier open-weight models and auto-generate summary metrics
python3 src/labeler.py --models main --summarize

# Evaluate thinking ablation checkpoints
python3 src/labeler.py --models ablation --summarize

# Evaluate zero-shot transformer encoders
python3 src/labeler.py --models encoders --summarize

# Run a quick 5-sample dry run on a specific model
python3 src/labeler.py --models gemma4:12b --max-samples 5
```

The runner automatically skips existing outputs unless `--overwrite` is specified, verifies 100% VRAM allocation, and unloads model weights after each checkpoint completes.

### Regenerating metric summaries
To re-aggregate raw predictions from `results/raw/` and `results/encoders/` into `results/summarized/`:
```bash
python3 src/summarize.py
```

### Visualizing Pareto curves and tables
Inspect quality vs. throughput tradeoffs, format adherence audits, and architectural rankings in the dedicated notebook [notebooks/benchmark_analysis.ipynb](file:///home/gsscala/code/llm-comparison/notebooks/benchmark_analysis.ipynb):
```bash
jupyter notebook notebooks/benchmark_analysis.ipynb
```
