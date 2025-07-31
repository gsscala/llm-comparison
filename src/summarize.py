"""Summarize raw prediction JSON files into standardized metric reports."""

import json
import os
import glob
import sys

try:
    from src.metrics import (
        calc_error_metrics,
        calc_qwk,
        calc_mcc,
        calc_macrof1,
        calc_time_metrics,
    )
except ImportError:
    from metrics import (
        calc_error_metrics,
        calc_qwk,
        calc_mcc,
        calc_macrof1,
        calc_time_metrics,
    )

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
ENCODERS_DIR = os.path.join(PROJECT_ROOT, "results", "encoders")
SUMMARIZED_DIR = os.path.join(PROJECT_ROOT, "results", "summarized")


def load_raw_records(file_path):
    """Load records from either JSON Lines (.jsonl) or a standard JSON array (.json)."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.strip().startswith("["):
            return json.load(f)
        records = []
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
        return records


def process_llm_records(file_path):
    """Compute all evaluation metrics for a raw LLM inference output file."""
    stats = load_raw_records(file_path)

    mae, mse, ob1, acc, mmae, invalid_rate, valid_only_mmae = calc_error_metrics(stats)
    mf1, wf1 = calc_macrof1(stats)
    qwk = calc_qwk(stats)
    mcc = calc_mcc(stats)
    eel, p50_eel, p95_eel, ttft, rps, tps = calc_time_metrics(stats)

    vram_mib = None
    gpu_residency = None
    for item in stats[:10]:
        if "vram_used_mib" in item and item["vram_used_mib"] is not None:
            vram_mib = item["vram_used_mib"]
            gpu_residency = item.get("gpu_residency_pct")
            break

    summary = {
        "Accuracy": acc,
        "Off-by-One Accuracy": ob1,
        "Macro-F1 Score": mf1,
        "Weighted-F1 Score": wf1,
        "Matthews Correlation Coefficient": mcc,
        "Quadratic Weighted Kappa": qwk,
        "Macro-Average Mean Absolute Error": mmae,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Invalid Prediction Rate": invalid_rate,
        "Valid-Only MMAE": valid_only_mmae,
        "Average end-to-end latency": eel,
        "Median end-to-end latency": p50_eel,
        "P95 end-to-end latency": p95_eel,
        "Average time to first token": ttft,
        "Requests per second": rps,
        "Output tokens per second": tps,
        "VRAM Used MiB": vram_mib,
        "GPU Residency Pct": gpu_residency,
    }
    return summary


def process_encoder_records(file_path):
    """Compute metrics for a transformer zero-shot encoder output file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    pred_true_pairs = content.get("prediction/true", [])
    if not pred_true_pairs:
        return None

    stats = [{"actual_label": p[0], "expected_label": p[1]} for p in pred_true_pairs]
    mae, mse, ob1, acc, mmae, invalid_rate, valid_only_mmae = calc_error_metrics(stats)
    mf1, wf1 = calc_macrof1(stats)
    qwk = calc_qwk(stats)
    mcc = calc_mcc(stats)

    raw_latency = content.get("Average end-to-end latency", 0.0)
    latency_ms = raw_latency * 1000.0 if (0.0 < raw_latency < 1.0) else raw_latency
    rps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

    summary = {
        "Accuracy": acc,
        "Off-by-One Accuracy": ob1,
        "Macro-F1 Score": mf1,
        "Weighted-F1 Score": wf1,
        "Matthews Correlation Coefficient": mcc,
        "Quadratic Weighted Kappa": qwk,
        "Macro-Average Mean Absolute Error": mmae,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Invalid Prediction Rate": invalid_rate,
        "Valid-Only MMAE": valid_only_mmae,
        "Average end-to-end latency": latency_ms,
        "Median end-to-end latency": latency_ms,
        "P95 end-to-end latency": latency_ms,
        "Requests per second": rps,
        "Output tokens per second": 0.0,
        "VRAM Used MiB": None,
        "GPU Residency Pct": 100.0,
    }
    return summary


def main(raw_dir=RAW_DIR, encoders_dir=ENCODERS_DIR, out_dir=SUMMARIZED_DIR):
    """Generate summary metric files from raw and encoder predictions."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating unified summaries in {out_dir}...")

    # 1. Process LLM runs
    if os.path.exists(raw_dir):
        for fname in sorted(os.listdir(raw_dir)):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            src_path = os.path.join(raw_dir, fname)
            if not os.path.isfile(src_path):
                continue
            try:
                summary = process_llm_records(src_path)
                if fname.endswith(".jsonl"):
                    out_name = f"{fname[:-6]}.json"
                elif fname.endswith(".json"):
                    out_name = fname
                else:
                    out_name = f"{fname}.json"
                dst_path = os.path.join(out_dir, out_name)
                with open(dst_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4)
                print(f"  [LLM] Summarized {fname} -> {out_name}")
            except Exception as e:
                print(f"  [LLM] Error processing {fname}: {e}")

    # 2. Process zero-shot encoders
    if os.path.exists(encoders_dir):
        for pattern in ["*_sst.json", "*_semeval.json"]:
            for src_path in glob.glob(os.path.join(encoders_dir, pattern)):
                fname = os.path.basename(src_path)
                try:
                    summary = process_encoder_records(src_path)
                    if summary:
                        dst_path = os.path.join(out_dir, fname)
                        with open(dst_path, "w", encoding="utf-8") as f:
                            json.dump(summary, f, indent=4)
                        print(f"  [Encoder] Summarized {fname}")
                except Exception as e:
                    print(f"  [Encoder] Error processing {fname}: {e}")


if __name__ == "__main__":
    main()