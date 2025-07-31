"""Zero-shot encoder benchmarking for modern NLI and GLiClass transformer models."""

import os
import sys
import json
import time
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import pipeline
from imblearn.metrics import macro_averaged_mean_absolute_error
from sklearn.metrics import accuracy_score

try:
    from src.config import ENCODER_MODELS
except ImportError:
    from config import ENCODER_MODELS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ENCODERS_DIR = os.path.join(PROJECT_ROOT, "results", "encoders")

CANDIDATE_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
LABEL_TO_INT = {lbl: idx for idx, lbl in enumerate(CANDIDATE_LABELS)}
GPU_DEVICE = int(os.environ.get("CUDA_DEVICE", 0)) if torch.cuda.is_available() else -1


def load_datasets():
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

    sst = pd.DataFrame(load_dataset("SetFit/sst5")["test"])
    sst = sst.drop(columns=["label_text"])

    return {"sst": sst, "semeval": semeval}


def run_encoder_evaluation(dataset_names=None, output_dir=ENCODERS_DIR):
    """Run zero-shot evaluation for all registered encoder models."""
    os.makedirs(output_dir, exist_ok=True)
    datasets = load_datasets()

    selected_datasets = (
        {k: v for k, v in datasets.items() if k in dataset_names}
        if dataset_names
        else datasets
    )

    for dataset_name, df in selected_datasets.items():
        print(f"\n{'='*50}\nEvaluating Dataset: {dataset_name}\n{'='*50}")

        hf_dataset = Dataset.from_pandas(df)
        if dataset_name == "semeval":
            hf_dataset = hf_dataset.map(lambda ex: {"text": f"Topic: {ex['hashtag']} | Tweet: {ex['text']}"})

        texts = list(hf_dataset["text"])
        clean_true_labels = [int(lbl) for lbl in hf_dataset["label"]]

        for enc in ENCODER_MODELS:
            model_id = enc["hf_id"]
            short_name = enc.get("short_name", model_id.split("/")[-1])
            enc_type = enc.get("encoder_type", "nli")
            print(f"\n--- Running Zero-Shot Inference with {short_name} ({enc_type}) on {dataset_name} ---")

            try:
                if enc_type == "gliclass":
                    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = GLiClassModel.from_pretrained(model_id)
                    device_str = f"cuda:{GPU_DEVICE}" if GPU_DEVICE >= 0 else "cpu"
                    classifier = ZeroShotClassificationPipeline(
                        model, tokenizer, classification_type="single-label", device=device_str
                    )

                    # Warmup pass (10 examples)
                    _ = classifier(texts[:10], CANDIDATE_LABELS, batch_size=8)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()
                    raw_preds = classifier(texts, CANDIDATE_LABELS, batch_size=1)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    predicted_ints = [LABEL_TO_INT[p[0]["label"]] for p in raw_preds]
                else:
                    classifier = pipeline("zero-shot-classification", model=model_id, device=GPU_DEVICE)

                    # Warmup pass (10 examples)
                    _ = classifier(texts[:10], candidate_labels=CANDIDATE_LABELS, batch_size=8)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start_time = time.perf_counter()
                    predictions = classifier(texts, candidate_labels=CANDIDATE_LABELS, batch_size=1)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    predicted_ints = [LABEL_TO_INT[pred["labels"][0]] for pred in predictions]

                instances = []
                for inp_text, pred, true in zip(texts, predicted_ints, clean_true_labels):
                    instances.append({
                        "instance": inp_text,
                        "expected_label": true,
                        "actual_label": pred,
                        "is_correct": bool(pred == true),
                        "ordinal_error": abs(pred - true),
                    })

                results = {
                    "Accuracy": accuracy_score(clean_true_labels, predicted_ints),
                    "Macro-Average Mean Absolute Error": macro_averaged_mean_absolute_error(
                        clean_true_labels, predicted_ints
                    ),
                    "Average end-to-end latency": elapsed_ms / len(texts),  # in milliseconds
                    "prediction/true": list(zip(predicted_ints, clean_true_labels)),
                    "instances": instances,
                }

                file_name = os.path.join(output_dir, f"{short_name}_{dataset_name}.json")
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)

                print(f"Saved results to {file_name}")

                del classifier
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                print(f"Error evaluating {short_name}: {exc}")


if __name__ == "__main__":
    run_encoder_evaluation()
