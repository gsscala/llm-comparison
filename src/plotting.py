"""Auxiliary visualization module for benchmark results, Pareto frontiers, and styling."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

try:
    from src.config import get_model_metadata
except ImportError:
    try:
        from config import get_model_metadata
    except ImportError:
        def get_model_metadata(name):
            clean = name.lower()
            if "encoder" in clean or any(e in clean for e in ["bert", "deberta", "bge", "gliclass"]):
                return {"family": "Encoder", "arch": "Encoder", "params_b": 0.35}
            if "moe" in clean or "26b" in clean or "35b" in clean or "flash" in clean:
                return {"family": clean.split("_")[0].capitalize(), "arch": "MoE", "params_b": 26.0}
            return {"family": clean.split("_")[0].capitalize(), "arch": "Dense", "params_b": 8.0}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SUMMARIZED_DIR = os.path.join(PROJECT_ROOT, "results", "summarized")

ARCH_MARKERS = {
    "Dense": "o",
    "MoE": "D",
    "Hybrid": "h",
    "Hybrid MoE": "d",
    "Encoder": "s",
}

FAMILY_PALETTE = {
    "Gemma 4": "#1f77b4",
    "Gemma (Legacy)": "#aec7e8",
    "Qwen 3.5/3.8": "#ff7f0e",
    "Qwen (Legacy)": "#ffbb78",
    "Llama / Muse": "#2ca02c",
    "gpt-oss": "#17becf",
    "Mistral 3": "#9467bd",
    "IBM Granite 4.2": "#8c564b",
    "Granite 4.2": "#8c564b",
    "Nemotron 3": "#7f7f7f",
    "Liquid LFM 2.5": "#bcbd22",
    "LFM 2.5": "#bcbd22",
    "AI2 OLMo 3 / 3.1": "#e377c2",
    "OLMo 3": "#e377c2",
    "Phi 4": "#d62728",
    "DeepSeek R1": "#393b79",
    "Aya Expanse": "#8c6d31",
    "GLM 4": "#bd9e39",
    "Encoder (NLI)": "#000000",
    "Encoder (GLiClass)": "#444444",
    "Encoder": "#000000",
}

_family_color_cache = dict(FAMILY_PALETTE)


def get_color_for_family(family: str) -> str:
    """Return a consistent hex color for a given model family."""
    if family in _family_color_cache:
        return _family_color_cache[family]
    h = (len(_family_color_cache) * 0.618033988749895) % 1.0
    col = mcolors.to_hex(mcolors.hsv_to_rgb((h, 0.7, 0.85)))
    _family_color_cache[family] = col
    return col


def adjust_text_safe(texts, x=None, y=None, ax=None):
    """Safely invoke adjust_text to minimize label collisions."""
    if adjust_text is None or not texts:
        return
    try:
        kwargs = {
            "ax": ax,
            "arrowprops": dict(arrowstyle="->", color="#666666", lw=0.6, alpha=0.65),
            "expand": (1.2, 1.4),
            "force_text": (0.7, 1.0),
            "force_static": (0.5, 0.7),
            "ensure_inside_axes": True,
        }
        if x is not None and y is not None:
            kwargs["x"] = x
            kwargs["y"] = y
        adjust_text(texts, **kwargs)
    except Exception:
        pass


def pareto_frontier_indices(xs, ys, maximize_x=True, maximize_y=True):
    """
    Find indices of points on the Pareto frontier.
    xs: throughput (higher is better)
    ys: accuracy (higher is better) or MMAE (lower is better)
    """
    if not xs or not ys or len(xs) != len(ys):
        return []
    pts = list(enumerate(zip(xs, ys)))
    pts.sort(key=lambda iv: iv[1][0], reverse=maximize_x)
    frontier = []
    if maximize_y:
        best_y = float("-inf")
        for idx, (x, y) in pts:
            if y >= best_y:
                frontier.append(idx)
                best_y = y
    else:
        best_y = float("inf")
        for idx, (x, y) in pts:
            if y <= best_y:
                frontier.append(idx)
                best_y = y
    return sorted(frontier, key=lambda i: xs[i])


def get_model_style(model_name: str) -> dict:
    """Return visual attributes (color, marker, family, architecture) for a model."""
    meta = get_model_metadata(model_name)
    family = meta.get("family", "Other")
    arch = meta.get("arch", "Dense")
    color = get_color_for_family(family)
    marker = ARCH_MARKERS.get(arch, "o")
    return {
        "family": family,
        "arch": arch,
        "color": color,
        "marker": marker,
        "params_b": meta.get("params_b"),
    }


def load_summary_dataframe(summary_dir=DEFAULT_SUMMARIZED_DIR) -> pd.DataFrame:
    """
    Load all JSON summary reports into a unified pandas DataFrame with metadata.
    """
    if not os.path.exists(summary_dir):
        return pd.DataFrame()

    records = []
    for file in sorted(os.listdir(summary_dir)):
        if not file.endswith(".json"):
            continue
        path = os.path.join(summary_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                perf = json.load(f)
        except Exception:
            continue

        is_sst = file.endswith("_sst.json")
        model_name = file[:-9] if is_sst else file[:-13]
        dataset_name = "SST-5" if is_sst else "SemEval"

        style = get_model_style(model_name)
        lat = perf.get("Average end-to-end latency", 0.0)
        tput = (1000.0 / lat) if lat > 0 else 0.0

        records.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Family": style["family"],
            "Arch": style["arch"],
            "Params_B": style["params_b"],
            "Accuracy": perf.get("Accuracy", 0.0),
            "Off_By_One": perf.get("Off-by-One Accuracy", 0.0),
            "Macro_F1": perf.get("Macro-F1 Score", 0.0),
            "Weighted_F1": perf.get("Weighted-F1 Score", 0.0),
            "MMAE": perf.get("Macro-Average Mean Absolute Error", 0.0),
            "MAE": perf.get("Mean Absolute Error", 0.0),
            "MSE": perf.get("Mean Squared Error", 0.0),
            "QWK": perf.get("Quadratic Weighted Kappa", 0.0),
            "MCC": perf.get("Matthews Correlation Coefficient", 0.0),
            "Latency_ms": lat,
            "Throughput": tput,
            "Invalid_Rate": perf.get("Invalid Prediction Rate", 0.0),
            "VRAM_MiB": perf.get("VRAM Used MiB"),
            "GPU_Residency_Pct": perf.get("GPU Residency Pct"),
        })

    return pd.DataFrame(records)


def get_architecture_legend_handles(include_encoders: bool = False):
    """Return standard matplotlib handles for architecture markers."""
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="Dense LLM"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="MoE LLM"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="Hybrid (Mamba/Conv)"),
        Line2D([0], [0], marker="d", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="Hybrid MoE"),
    ]
    if include_encoders:
        handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="Zero-Shot Encoder")
        )
    handles.append(
        Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor="k", markersize=10, markeredgewidth=1.8, label="Pareto Efficient")
    )
    return handles


def get_encoder_legend_handles():
    """Return standard matplotlib handles for zero-shot encoder markers."""
    return [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1d4ed8", markeredgecolor="k", markersize=8, label="ModernBERT"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a", markeredgecolor="k", markersize=8, label="DeBERTa-v3"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ea580c", markeredgecolor="k", markersize=8, label="BGE-M3"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9333ea", markeredgecolor="k", markersize=8, label="GLiClass"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#64748b", markeredgecolor="k", markersize=8, label="Baselines (RoBERTa / BART)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor="k", markersize=10, markeredgewidth=1.8, label="Pareto Efficient"),
    ]


ENCODER_MODEL_INFO = {
    "ModernBERT-base-zeroshot-v2.0": {"display": "ModernBERT-base", "color": "#0284c7"},
    "ModernBERT-large-zeroshot-v2.0": {"display": "ModernBERT-large", "color": "#1d4ed8"},
    "deberta-v3-large-zeroshot-v2.0": {"display": "DeBERTa-v3-large", "color": "#16a34a"},
    "bge-m3-zeroshot-v2.0": {"display": "BGE-M3", "color": "#ea580c"},
    "gliclass-base-v1.0": {"display": "GLiClass-base", "color": "#9333ea"},
    "gliclass-large-v1.0": {"display": "GLiClass-large", "color": "#4f46e5"},
    "RoBERTa": {"display": "RoBERTa (baseline)", "color": "#64748b"},
    "BART": {"display": "BART (baseline)", "color": "#b45309"},
}


def plot_benchmark_frontier(
    df: pd.DataFrame,
    metric: str = "MMAE",
    figsize=(10, 18),
    include_encoders: bool = False,
    save_path=None,
):
    """
    Plot Pareto frontier subplots for SST-5 and SemEval across evaluated generative LLMs.
    metric: 'MMAE' (lower is better, inverted y-axis) or 'Accuracy' (higher is better).
    """
    if df.empty:
        print("Warning: DataFrame is empty, nothing to plot.")
        return None

    # Exclude encoders by default so their 10x-40x higher throughput does not compress LLM spacing
    source_df = df.copy()
    if not include_encoders and "Arch" in source_df.columns:
        source_df = source_df[source_df["Arch"] != "Encoder"].copy()

    is_mmae = (metric.upper() == "MMAE")
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    datasets = [
        ("SST-5", "SST Dataset (SST-5)"),
        ("SemEval", "SemEval Dataset (Task 4B)"),
    ]

    for ax, (dkey, dtitle) in zip(axes, datasets):
        sub_df = source_df[source_df["Dataset"] == dkey].copy()
        if sub_df.empty:
            continue

        # Filter out runs that completely failed token output (e.g. 100% format failure)
        plot_df = sub_df[sub_df["Invalid_Rate"] < 0.95].copy() if "Invalid_Rate" in sub_df.columns else sub_df.copy()
        if plot_df.empty:
            plot_df = sub_df.copy()

        ax.set_xlabel("Instances per second (Throughput)", fontsize=15)
        ax.set_title(dtitle, fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Dataset-adaptive y-axis bounds
        valid_ys = plot_df["MMAE" if is_mmae else "Accuracy"].dropna().tolist()
        if valid_ys:
            y_min = min(valid_ys)
            y_max = max(valid_ys)
            y_span = max(0.05, y_max - y_min)
            pad = max(0.02, y_span * 0.08)
            if is_mmae:
                ax.set_ylabel("Macro-Average Mean Absolute Error (Lower is Better)", fontsize=15)
                y_best = max(0.0, round(y_min - pad, 2))
                y_worst = round(y_max + pad, 2)
                if abs(y_worst - y_best) < 0.04:
                    y_best = max(0.0, y_best - 0.03)
                    y_worst += 0.03
                ax.set_ylim(bottom=y_worst, top=y_best)
            else:
                ax.set_ylabel("Accuracy (Higher is Better)", fontsize=15)
                y_bottom = max(0.0, round(y_min - pad, 2))
                y_top = min(1.0, round(y_max + pad, 2))
                if abs(y_top - y_bottom) < 0.04:
                    y_bottom = max(0.0, y_bottom - 0.03)
                    y_top = min(1.0, y_top + 0.03)
                ax.set_ylim(bottom=y_bottom, top=y_top)

        xs, ys, texts = [], [], []
        for _, row in plot_df.iterrows():
            m_name = row["Model"]
            style = get_model_style(m_name)
            x = row["Throughput"]
            y = row["MMAE"] if is_mmae else row["Accuracy"]

            gpu_res = row.get("GPU_Residency_Pct")
            edge_col = "#ff0055" if (gpu_res is not None and gpu_res < 99.0) else "black"
            edge_w = 1.8 if (gpu_res is not None and gpu_res < 99.0) else 0.6

            ax.scatter(
                x, y,
                color=style["color"],
                marker=style["marker"],
                s=70,
                edgecolors=edge_col,
                linewidths=edge_w,
                zorder=3,
            )

            inv_rate = row.get("Invalid_Rate", 0.0)
            label_str = f"{m_name}*" if inv_rate > 0.15 else m_name
            t = ax.annotate(
                label_str,
                (x, y),
                fontsize=8.5,
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
            texts.append(t)
            xs.append(x)
            ys.append(y)

        adjust_text_safe(texts, x=xs, y=ys, ax=ax)

        # Compute Pareto frontier
        p_idx = pareto_frontier_indices(xs, ys, maximize_x=True, maximize_y=(not is_mmae))
        if p_idx:
            frontier_x = [xs[i] for i in p_idx]
            frontier_y = [ys[i] for i in p_idx]
            ax.plot(frontier_x, frontier_y, color="black", linestyle=":", lw=1.5, zorder=4, label="Pareto Frontier")
            ax.scatter(frontier_x, frontier_y, marker="^", color="none", edgecolors="black", s=120, lw=1.8, zorder=5)

        ax.legend(
            handles=get_architecture_legend_handles(include_encoders=include_encoders),
            loc="lower left",
            fontsize=11,
            framealpha=0.9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def plot_encoder_frontier(
    df: pd.DataFrame,
    metric: str = "MMAE",
    figsize=(10, 14),
    save_path=None,
):
    """
    Plot Pareto frontier subplots for SST-5 and SemEval across zero-shot transformer encoders.
    metric: 'MMAE' (lower is better, inverted y-axis) or 'Accuracy' (higher is better).
    """
    if df.empty:
        print("Warning: DataFrame is empty, nothing to plot.")
        return None

    source_df = df[df["Arch"] == "Encoder"].copy() if "Arch" in df.columns else df.copy()
    if source_df.empty:
        print("Warning: No encoder models found in DataFrame.")
        return None

    is_mmae = (metric.upper() == "MMAE")
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    datasets = [
        ("SST-5", "Zero-Shot Encoders on SST-5"),
        ("SemEval", "Zero-Shot Encoders on SemEval (Task 4B)"),
    ]

    for ax, (dkey, dtitle) in zip(axes, datasets):
        sub_df = source_df[source_df["Dataset"] == dkey].copy()
        if sub_df.empty:
            continue

        ax.set_xlabel("Instances per second (Throughput)", fontsize=15)
        ax.set_title(dtitle, fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Dataset-adaptive y-axis bounds
        valid_ys = sub_df["MMAE" if is_mmae else "Accuracy"].dropna().tolist()
        if valid_ys:
            y_min = min(valid_ys)
            y_max = max(valid_ys)
            y_span = max(0.04, y_max - y_min)
            pad = max(0.02, y_span * 0.12)
            if is_mmae:
                ax.set_ylabel("Macro-Average Mean Absolute Error (Lower is Better)", fontsize=15)
                y_best = max(0.0, round(y_min - pad, 2))
                y_worst = round(y_max + pad, 2)
                if abs(y_worst - y_best) < 0.04:
                    y_best = max(0.0, y_best - 0.03)
                    y_worst += 0.03
                ax.set_ylim(bottom=y_worst, top=y_best)
            else:
                ax.set_ylabel("Accuracy (Higher is Better)", fontsize=15)
                y_bottom = max(0.0, round(y_min - pad, 2))
                y_top = min(1.0, round(y_max + pad, 2))
                if abs(y_top - y_bottom) < 0.04:
                    y_bottom = max(0.0, y_bottom - 0.03)
                    y_top = min(1.0, y_top + 0.03)
                ax.set_ylim(bottom=y_bottom, top=y_top)

        # Adaptive x-axis bounds with room for annotations
        valid_xs = sub_df["Throughput"].dropna().tolist()
        if valid_xs:
            ax.set_xlim(left=0, right=max(valid_xs) * 1.15)

        xs, ys, texts = [], [], []
        for _, row in sub_df.iterrows():
            m_name = row["Model"]
            info = ENCODER_MODEL_INFO.get(m_name, {
                "display": m_name,
                "color": "#333333",
            })
            x = row["Throughput"]
            y = row["MMAE"] if is_mmae else row["Accuracy"]

            ax.scatter(
                x, y,
                color=info["color"],
                marker="s",
                s=90,
                edgecolors="black",
                linewidths=0.9,
                zorder=3,
            )

            display_name = info["display"]
            t = ax.annotate(
                display_name,
                (x, y),
                fontsize=9.5,
                alpha=0.95,
                fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85, lw=0.6),
            )
            texts.append(t)
            xs.append(x)
            ys.append(y)

        adjust_text_safe(texts, x=xs, y=ys, ax=ax)

        # Compute Pareto frontier
        p_idx = pareto_frontier_indices(xs, ys, maximize_x=True, maximize_y=(not is_mmae))
        if p_idx:
            frontier_x = [xs[i] for i in p_idx]
            frontier_y = [ys[i] for i in p_idx]
            ax.plot(frontier_x, frontier_y, color="black", linestyle=":", lw=1.6, zorder=4, label="Pareto Frontier")
            ax.scatter(frontier_x, frontier_y, marker="^", color="none", edgecolors="black", s=130, lw=2.0, zorder=5)

        legend_loc = "lower left" if is_mmae else "lower right"
        ax.legend(handles=get_encoder_legend_handles(), loc=legend_loc, fontsize=10.5, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def plot_encoder_comparison(
    df: pd.DataFrame,
    dataset: str = "SST-5",
    figsize=(13, 4.5),
    save_path=None,
):
    """Plot bar charts comparing zero-shot encoder models on accuracy, MMAE, and throughput."""
    if df.empty or "Dataset" not in df.columns:
        return None

    sub = df[(df["Dataset"] == dataset) & (df["Arch"] == "Encoder")].copy()
    if sub.empty:
        return None

    sub = sub.sort_values("Throughput", ascending=False).reset_index(drop=True)
    sub["DisplayName"] = sub["Model"].apply(lambda m: ENCODER_MODEL_INFO.get(m, {}).get("display", m))
    colors = [ENCODER_MODEL_INFO.get(m, {}).get("color", "#333333") for m in sub["Model"]]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(sub["DisplayName"], sub["Accuracy"], color=colors, alpha=0.85, edgecolor="k")
    axes[0].set_title(f"Accuracy ({dataset})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (Higher is Better)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].bar(sub["DisplayName"], sub["MMAE"], color=colors, alpha=0.85, edgecolor="k")
    axes[1].set_title(f"MMAE ({dataset})", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MMAE (Lower is Better)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].bar(sub["DisplayName"], sub["Throughput"], color=colors, alpha=0.85, edgecolor="k")
    axes[2].set_title(f"Throughput ({dataset})", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Instances / sec")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def audit_raw_predictions(raw_dir: str | None = None) -> pd.DataFrame:
    """Audit all raw prediction files in results/raw for formatting failures."""
    import re
    try:
        from src.predictor import parse_sentiment_to_id
    except ImportError:
        from predictor import parse_sentiment_to_id

    target_dir = raw_dir or os.path.join(PROJECT_ROOT, "results", "raw")
    if not os.path.exists(target_dir):
        return pd.DataFrame()

    audit_rows = []
    for fname in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first_line = f.readline()
                f.seek(0)
                if first_line.strip().startswith("["):
                    instances = json.load(f)
                else:
                    instances = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue

        total = len(instances)
        invalid_cnt = 0
        multi_num_cnt = 0
        no_num_cnt = 0

        for inst in instances:
            pred = inst.get("actual_label", inst.get("predicted_label"))
            raw = inst.get("raw_model_output", "")
            if pred is None or pred < 0 or pred > 4:
                pred = parse_sentiment_to_id(raw)

            if pred < 0:
                invalid_cnt += 1
                clean_body = raw.split("</think>", 1)[1] if "</think>" in str(raw) else str(raw)
                digits = re.findall(r"\b[0-4]\b", clean_body)
                if len(set(digits)) > 1:
                    multi_num_cnt += 1
                else:
                    no_num_cnt += 1

        is_sst = fname.endswith("_sst.jsonl") or fname.endswith("_sst.json") or fname.endswith("_sst")
        dataset = "SST-5" if is_sst else "SemEval"
        clean_tag = (
            fname.replace(".jsonl", "")
            .replace(".json", "")
            .replace("_sst", "")
            .replace("_semeval", "")
        )

        audit_rows.append({
            "Model": clean_tag,
            "Dataset": dataset,
            "Total_Samples": total,
            "Invalid_Count": invalid_cnt,
            "Invalid_Rate_Pct": round(100.0 * invalid_cnt / total, 2) if total else 0.0,
            "Multiple_Digits": multi_num_cnt,
            "No_Digits_Found": no_num_cnt,
        })

    return pd.DataFrame(audit_rows).sort_values("Invalid_Rate_Pct", ascending=False).reset_index(drop=True)


def format_comparison_table(
    df: pd.DataFrame,
    dataset: str = "SST-5",
    sort_by: str = "Accuracy",
    ascending: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return a clean formatted summary DataFrame for notebook display."""
    if df.empty or "Dataset" not in df.columns:
        return pd.DataFrame()

    sub = df[df["Dataset"] == dataset].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    cols = [
        "Model", "Family", "Arch", "Params_B",
        "Accuracy", "Off_By_One", "MMAE", "QWK",
        "Throughput", "Latency_ms", "Invalid_Rate"
    ]
    available_cols = [c for c in cols if c in sub.columns]
    if top_n == None:
        top_n = len(sub[available_cols])
    res = sub[available_cols].head(top_n).copy()

    # Round floating columns for presentation
    for float_col in ["Accuracy", "Off_By_One", "MMAE", "QWK"]:
        if float_col in res.columns:
            res[float_col] = res[float_col].round(4)
    if "Throughput" in res.columns:
        res["Throughput"] = res["Throughput"].round(2)
    if "Latency_ms" in res.columns:
        res["Latency_ms"] = res["Latency_ms"].round(1)
    if "Invalid_Rate" in res.columns:
        res["Invalid_Rate_Pct"] = (res["Invalid_Rate"] * 100).round(2)
        res = res.drop(columns=["Invalid_Rate"])

    return res


def plot_architecture_comparison(
    df: pd.DataFrame,
    dataset: str = "SST-5",
    figsize=(12, 4),
    include_encoders: bool = False,
    save_path=None,
):
    """Plot bar chart comparing mean accuracy, MMAE, and throughput by architecture."""
    if df.empty or "Dataset" not in df.columns:
        return None

    sub = df[df["Dataset"] == dataset].copy()
    if sub.empty:
        return None

    # Exclude encoders by default so their throughput does not compress generative LLM bars
    if not include_encoders and "Arch" in sub.columns:
        sub = sub[sub["Arch"] != "Encoder"].copy()

    if sub.empty:
        return None

    if "Invalid_Rate" in sub.columns:
        valid_sub = sub[sub["Invalid_Rate"] < 0.95].copy()
        if not valid_sub.empty:
            sub = valid_sub

    grouped = sub.groupby("Arch").agg(
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_MMAE=("MMAE", "mean"),
        Mean_Throughput=("Throughput", "mean"),
        Count=("Model", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(grouped["Arch"], grouped["Mean_Accuracy"], color="#2ca02c", alpha=0.85, edgecolor="k")
    axes[0].set_title(f"Mean Accuracy ({dataset})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (Higher is Better)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].bar(grouped["Arch"], grouped["Mean_MMAE"], color="#d62728", alpha=0.85, edgecolor="k")
    axes[1].set_title(f"Mean MMAE ({dataset})", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MMAE (Lower is Better)")
    axes[1].set_ylim(bottom=0.0, top=1.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].bar(grouped["Arch"], grouped["Mean_Throughput"], color="#1f77b4", alpha=0.85, edgecolor="k")
    axes[2].set_title(f"Mean Throughput ({dataset})", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Instances / sec")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
