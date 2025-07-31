"""Dedicated labeling and inference runner for open-weight LLMs and zero-shot encoders."""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import pandas as pd
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
SUMMARIZED_DIR = os.path.join(PROJECT_ROOT, "results", "summarized")
ENCODERS_DIR = os.path.join(PROJECT_ROOT, "results", "encoders")

from src.config import (
    MAIN_LLM_MODELS,
    THINKING_ABLATION_MODELS,
    ENCODER_MODELS,
    PROMPT_JSON_SST5,
    PROMPT_JSON_SEMEVAL,
    PROMPT_THINK_SST5,
    PROMPT_THINK_SEMEVAL,
)
from src.predictor import Predictor
from src.summarize import main as run_summarize


def get_encoder_evaluator():
    """Lazily import encoder evaluator to avoid requiring torch on minimal environments."""
    try:
        from src.encoders import run_encoder_evaluation
        return run_encoder_evaluation
    except ImportError as e:
        raise ImportError(f"PyTorch and Transformers are required for encoder evaluation: {e}")


def load_benchmark_datasets() -> Dict[str, pd.DataFrame]:
    """Load and format SST-5 and SemEval-2017 Task 4B datasets."""
    semeval_path = os.path.join(DATA_DIR, "semeval.txt")
    if not os.path.exists(semeval_path):
        raise FileNotFoundError(f"SemEval dataset not found at {semeval_path}")

    semeval = pd.read_csv(
        semeval_path,
        sep="\t",
        header=None,
        names=["tweet_id", "hashtag", "label", "text"],
        encoding="utf-8",
        quoting=3,
    )
    semeval["label"] = semeval["label"].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
    semeval = semeval.drop(columns=["tweet_id"])

    if load_dataset is not None:
        sst = pd.DataFrame(load_dataset("SetFit/sst5")["test"])
        sst = sst.drop(columns=["label_text"])
    else:
        sst_local = os.path.join(DATA_DIR, "sst5.csv")
        if os.path.exists(sst_local):
            sst = pd.read_csv(sst_local)
        else:
            raise ImportError(
                "The 'datasets' package is required to download SST-5. "
                "Install it with 'pip install datasets' or place 'sst5.csv' in data/."
            )

    return {"sst": sst, "semeval": semeval}


def format_dataset_input(dataset_name: str, row: pd.Series) -> str:
    """Format single row into task-appropriate input string."""
    if dataset_name == "sst":
        return str(row["text"])
    return f"Topic: {row['hashtag']}\nTweet: {row['text']}"


def get_raw_output_path(output_dir: str, safe_name: str, dname: str) -> str:
    """Return output path, checking for existing .jsonl or .json files, defaulting to .jsonl."""
    jsonl_path = os.path.join(output_dir, f"{safe_name}_{dname}.jsonl")
    json_path = os.path.join(output_dir, f"{safe_name}_{dname}.json")
    if os.path.exists(jsonl_path):
        return jsonl_path
    if os.path.exists(json_path):
        return json_path
    return jsonl_path


def run_single_model_labeling(
    model_tag: str,
    think: bool = False,
    datasets: Optional[Dict[str, pd.DataFrame]] = None,
    dataset_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    output_dir: str = RAW_DIR,
    host: Optional[str] = None,
    client=None,
) -> Dict[str, str]:
    """
    Run labeling for a single model across requested datasets.
    Returns mapping of dataset name to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    if datasets is None:
        datasets = load_benchmark_datasets()

    target_datasets = dataset_names or ["sst", "semeval"]
    safe_name = model_tag.replace(":", "_") + ("_think" if think else "")
    out_paths = {}

    # Check if all targets already exist to avoid loading model weights unnecessarily
    all_exist = True
    for dname in target_datasets:
        target_path = get_raw_output_path(output_dir, safe_name, dname)
        if not os.path.exists(target_path):
            all_exist = False
            break

    if all_exist and not overwrite:
        print(f"Skipping {safe_name}: all target outputs already exist in {output_dir}.")
        return {dname: get_raw_output_path(output_dir, safe_name, dname) for dname in target_datasets}

    predictor: Optional[Predictor] = None
    vram_used_mib = None
    gpu_residency_pct = None

    for dname in target_datasets:
        if dname not in datasets:
            print(f"Warning: dataset '{dname}' not recognized, skipping.")
            continue

        existing_path = get_raw_output_path(output_dir, safe_name, dname)
        if os.path.exists(existing_path) and not overwrite:
            print(f"Skipping {safe_name} on {dname}: output already exists at {existing_path}.")
            out_paths[dname] = existing_path
            continue

        target_path = os.path.join(output_dir, f"{safe_name}_{dname}.jsonl")
        prompt = (
            (PROMPT_THINK_SST5 if think else PROMPT_JSON_SST5)
            if dname == "sst"
            else (PROMPT_THINK_SEMEVAL if think else PROMPT_JSON_SEMEVAL)
        )

        # Initialize predictor on first dataset run
        if predictor is None:
            print(f"\nInitializing {safe_name} (think={think}, temp=0.0) on GPU...")
            predictor = Predictor(
                model_name=model_tag,
                prompt_template=prompt,
                think=think,
                host=host,
                warmup=True,
                client=client,
            )
            vram_used_mib = predictor.vram_used_mib
            gpu_residency_pct = predictor.gpu_residency_pct
        else:
            predictor.prompt_template = prompt

        df = datasets[dname]
        if max_samples is not None:
            df = df.head(max_samples)

        print(f"Labeling {dname} ({len(df)} samples) with {safe_name}...")
        sample_count = 0
        with open(target_path, "w", encoding="utf-8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{safe_name} [{dname}]", leave=False):
                inp_text = format_dataset_input(dname, row)
                rec = predictor.predict(inp_text)
                expected = int(row["label"])
                actual = rec.get("actual_label", -1)
                rec["expected_label"] = expected
                rec["actual_label"] = actual
                rec["is_correct"] = bool(actual == expected) if actual >= 0 else False
                rec["ordinal_error"] = abs(actual - expected) if actual >= 0 else None
                rec["dataset"] = dname
                rec["model"] = model_tag
                # Attach VRAM metadata
                if vram_used_mib is not None:
                    rec["vram_used_mib"] = vram_used_mib
                    rec["gpu_residency_pct"] = gpu_residency_pct
                f.write(json.dumps(rec) + "\n")
                f.flush()
                sample_count += 1

        print(f"Saved {sample_count} predictions to {target_path}")
        out_paths[dname] = target_path

    # Unload model weights from GPU memory
    if predictor is not None:
        predictor.unload()
        print(f"Evicted weights for {model_tag} from VRAM.")

    return out_paths


def run_llm_benchmarks(
    model_entries: List[dict],
    think: Optional[bool] = None,
    dataset_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    output_dir: str = RAW_DIR,
    host: Optional[str] = None,
    client=None,
):
    """Iterate over multiple LLM model configs and run labeling."""
    datasets = load_benchmark_datasets()
    total_models = len(model_entries)
    print(f"\nStarting labeling suite across {total_models} model runs...")

    model_pbar = tqdm(model_entries, desc="Evaluating Models", unit="model", leave=True)
    for i, entry in enumerate(model_pbar, start=1):
        tag = entry["tag"]
        use_think = think if think is not None else entry.get("think", False)
        print(f"\n{'='*60}\n[{i}/{total_models}] Benchmark run: {tag} (think={use_think})\n{'='*60}")
        try:
            run_single_model_labeling(
                model_tag=tag,
                think=use_think,
                datasets=datasets,
                dataset_names=dataset_names,
                max_samples=max_samples,
                overwrite=overwrite,
                output_dir=output_dir,
                host=host,
                client=client,
            )
        except Exception as exc:
            print(f"Error executing labeling for {tag}: {exc}", file=sys.stderr)


def main():
    """Command-line entry point for running the labeling benchmark."""
    parser = argparse.ArgumentParser(
        description="Run local sentiment labeling across frontier open-weight models and zero-shot encoders."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="main",
        help="Model preset ('main', 'ablation', 'all-llms', 'encoders', 'all') or comma-separated tags.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated datasets ('sst', 'semeval', or 'all').",
    )
    parser.add_argument(
        "--think",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Thinking flag ('auto' respects model config, 'true' forces thinking, 'false' disables).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of instances per dataset for testing or quick dry-runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output prediction files.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Automatically generate summary metric JSONs after labeling completes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RAW_DIR,
        help="Directory to save raw model output JSON files.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama server URL (e.g. http://192.168.1.50:11434 or via SSH tunnel). Defaults to OLLAMA_HOST or localhost.",
    )

    args = parser.parse_args()

    # Resolve datasets
    if args.datasets == "all":
        dataset_names = ["sst", "semeval"]
    else:
        dataset_names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    # Resolve thinking flag
    think_mode = None
    if args.think == "true":
        think_mode = True
    elif args.think == "false":
        think_mode = False

    # Check if encoders are requested
    if args.models in ["encoders", "all"]:
        print("\n=== Running Zero-Shot Transformer Encoders ===")
        evaluator = get_encoder_evaluator()
        evaluator(dataset_names=dataset_names, output_dir=ENCODERS_DIR)

    # Resolve LLM models
    llm_entries = []
    if args.models == "main":
        llm_entries = MAIN_LLM_MODELS
    elif args.models == "ablation":
        llm_entries = THINKING_ABLATION_MODELS
    elif args.models in ["all-llms", "all"]:
        # Combine main and thinking ablation
        llm_entries = list(MAIN_LLM_MODELS)
        for m in THINKING_ABLATION_MODELS:
            m_copy = dict(m)
            m_copy["think"] = True
            llm_entries.append(m_copy)
    elif args.models != "encoders":
        tags = [t.strip() for t in args.models.split(",") if t.strip()]
        for t in tags:
            llm_entries.append({"tag": t, "think": bool(think_mode)})

    if llm_entries:
        run_llm_benchmarks(
            model_entries=llm_entries,
            think=think_mode,
            dataset_names=dataset_names,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
            output_dir=args.output_dir,
            host=args.host,
        )

    if args.summarize:
        print("\n=== Running Summary Aggregation ===")
        run_summarize(raw_dir=args.output_dir, encoders_dir=ENCODERS_DIR, out_dir=SUMMARIZED_DIR)

    print("\nBenchmark run completed successfully.")


if __name__ == "__main__":
    main()
