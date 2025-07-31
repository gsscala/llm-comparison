"""LLM Comparison: Benchmarking frontier open-weight models against zero-shot encoders."""

from src.config import (
    MAIN_LLM_MODELS,
    THINKING_ABLATION_MODELS,
    ENCODER_MODELS,
    VRAM_BUDGET_MIB,
    PROMPT_JSON_SST5,
    PROMPT_JSON_SEMEVAL,
    PROMPT_THINK_SST5,
    PROMPT_THINK_SEMEVAL,
    get_model_metadata,
)
from src.predictor import Predictor, parse_sentiment_to_id
from src.metrics import (
    extract_labels,
    calc_error_metrics,
    calc_qwk,
    calc_mcc,
    calc_macrof1,
    calc_time_metrics,
)
from src.summarize import process_llm_records, process_encoder_records
from src.labeler import (
    load_benchmark_datasets,
    run_single_model_labeling,
    run_llm_benchmarks,
)

from src.plotting import (
    load_summary_dataframe,
    plot_benchmark_frontier,
    audit_raw_predictions,
    format_comparison_table,
    plot_architecture_comparison,
    get_model_style,
    pareto_frontier_indices,
)

__all__ = [
    "MAIN_LLM_MODELS",
    "THINKING_ABLATION_MODELS",
    "ENCODER_MODELS",
    "VRAM_BUDGET_MIB",
    "PROMPT_JSON_SST5",
    "PROMPT_JSON_SEMEVAL",
    "PROMPT_THINK_SST5",
    "PROMPT_THINK_SEMEVAL",
    "get_model_metadata",
    "Predictor",
    "parse_sentiment_to_id",
    "extract_labels",
    "calc_error_metrics",
    "calc_qwk",
    "calc_mcc",
    "calc_macrof1",
    "calc_time_metrics",
    "process_llm_records",
    "process_encoder_records",
    "load_benchmark_datasets",
    "run_single_model_labeling",
    "run_llm_benchmarks",
    "load_summary_dataframe",
    "plot_benchmark_frontier",
    "audit_raw_predictions",
    "format_comparison_table",
    "plot_architecture_comparison",
    "get_model_style",
    "pareto_frontier_indices",
]
