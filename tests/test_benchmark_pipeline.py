"""Unit tests for the modernized benchmark pipeline."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import (
    MAIN_LLM_MODELS,
    THINKING_ABLATION_MODELS,
    ENCODER_MODELS,
    get_model_metadata,
    VRAM_BUDGET_MIB,
)
from src.predictor import parse_sentiment_to_id, Predictor
from src.metrics import (
    calc_error_metrics,
    calc_macrof1,
    calc_mcc,
    calc_qwk,
    calc_time_metrics,
)
from src.plotting import (
    pareto_frontier_indices,
    get_model_style,
    load_summary_dataframe,
    format_comparison_table,
    audit_raw_predictions,
    plot_benchmark_frontier,
    plot_encoder_frontier,
    plot_architecture_comparison,
    plot_encoder_comparison,
    adjust_text_safe,
)
from src.labeler import (
    format_dataset_input,
    run_single_model_labeling,
)
from src.summarize import process_llm_records
from src.pull_models import check_models_status, resolve_model_entries


class TestModelsConfig(unittest.TestCase):
    def test_model_counts_and_sizes(self):
        self.assertEqual(len(MAIN_LLM_MODELS), 30)
        self.assertEqual(len(THINKING_ABLATION_MODELS), 8)
        self.assertEqual(len(ENCODER_MODELS), 6)

        for m in MAIN_LLM_MODELS:
            self.assertIn("tag", m)
            self.assertIn("family", m)
            self.assertIn("arch", m)
            self.assertLessEqual(m["est_mib"], VRAM_BUDGET_MIB)

    def test_metadata_lookup(self):
        meta_gemma = get_model_metadata("gemma4_31b_sst")
        self.assertEqual(meta_gemma["family"], "Gemma 4")
        self.assertEqual(meta_gemma["arch"], "Dense")

        meta_qwen = get_model_metadata("qwen3.6_35b_semeval")
        self.assertEqual(meta_qwen["family"], "Qwen 3.5/3.8")
        self.assertEqual(meta_qwen["arch"], "MoE")

        meta_enc = get_model_metadata("ModernBERT_sst")
        self.assertEqual(meta_enc["arch"], "Encoder")


class TestPredictionParsing(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(parse_sentiment_to_id('{"sentiment": 4}'), 4)
        self.assertEqual(parse_sentiment_to_id('{"sentiment": 0}'), 0)
        self.assertEqual(parse_sentiment_to_id('{"sentiment": 2}'), 2)

    def test_json_with_surrounding_text(self):
        self.assertEqual(
            parse_sentiment_to_id('Here is your answer: {"sentiment": 3}\nHope this helps!'),
            3,
        )

    def test_think_tag_stripping(self):
        text = "<think>The movie had great acting so I think it is positive 3</think>\n{\"sentiment\": 3}"
        self.assertEqual(parse_sentiment_to_id(text), 3)

        text_free = "<think>The review is negative 1</think>\n1"
        self.assertEqual(parse_sentiment_to_id(text_free), 1)

    def test_unclosed_think_tag(self):
        text = "<think>Cutoff generation before closing tag..."
        self.assertEqual(parse_sentiment_to_id(text), -1)

    def test_ambiguous_output(self):
        text = "It could be 1 or perhaps 2 or 3"
        self.assertEqual(parse_sentiment_to_id(text), -1)

    def test_out_of_bounds_sentiment(self):
        self.assertEqual(parse_sentiment_to_id('{"sentiment": 5}'), -1)
        self.assertEqual(parse_sentiment_to_id('{"sentiment": -2}'), -1)


class TestPredictorWithMockClient(unittest.TestCase):
    def test_predictor_warmup_and_inference(self):
        mock_client = MagicMock()

        mock_client.ps.return_value = {
            "models": [
                {
                    "name": "gemma4:12b",
                    "model": "gemma4:12b",
                    "size": 8074035200,      # 7.7 GB
                    "size_vram": 8074035200, # 100% in VRAM
                }
            ]
        }

        mock_chunks = [
            {"message": {"content": '{"sentiment": '}},
            {"message": {"content": '4}'}, "done": True, "eval_count": 5, "eval_duration": 50000000, "prompt_eval_duration": 10000000, "total_duration": 65000000},
        ]
        mock_client.chat.return_value = mock_chunks

        predictor = Predictor(
            model_name="gemma4:12b",
            prompt_template='Classify: "{sentence}"',
            think=False,
            warmup=True,
            client=mock_client,
        )

        self.assertEqual(predictor.gpu_residency_pct, 100.0)
        self.assertAlmostEqual(predictor.vram_used_mib, 7700.0, delta=100.0)

        rec = predictor.predict("Fantastic cinematic experience!")
        self.assertEqual(rec["actual_label"], 4)
        self.assertEqual(rec["temperature"], 0.0)
        self.assertEqual(rec["seed"], 42)
        self.assertEqual(predictor.host, None)
        self.assertEqual(rec["output_token_count"], 5)
        self.assertEqual(rec["end_to_end_latency_ms"], 65.0)
        self.assertEqual(rec["time_to_first_token_ms"], 10.0)
        self.assertEqual(rec["generation_duration_ms"], 50.0)

        predictor.unload()
        mock_client.chat.assert_called_with(model="gemma4:12b", messages=[], keep_alive=0)


class TestMetricsUnbiasedCalculations(unittest.TestCase):
    def test_perfect_predictions(self):
        data = [{"actual_label": i, "expected_label": i} for i in range(5)] * 20
        mae, mse, ob1, acc, mmae, invalid_rate, valid_mmae = calc_error_metrics(data)
        self.assertEqual(acc, 1.0)
        self.assertEqual(ob1, 1.0)
        self.assertEqual(mae, 0.0)
        self.assertEqual(mse, 0.0)
        self.assertEqual(mmae, 0.0)
        self.assertEqual(invalid_rate, 0.0)

        qwk = calc_qwk(data)
        self.assertEqual(qwk, 1.0)

        mf1, wf1 = calc_macrof1(data)
        self.assertEqual(mf1, 1.0)
        self.assertEqual(wf1, 1.0)

        mcc = calc_mcc(data)
        self.assertEqual(mcc, 1.0)

    def test_invalid_predictions_penalty(self):
        data = []
        for i in range(5):
            data.append({"actual_label": i, "expected_label": i})   # correct
            data.append({"actual_label": -1, "expected_label": i})  # failed parse

        mae, mse, ob1, acc, mmae, invalid_rate, valid_mmae = calc_error_metrics(data)
        self.assertEqual(invalid_rate, 0.5)
        self.assertEqual(acc, 0.5)
        self.assertEqual(ob1, 0.5)
        self.assertEqual(valid_mmae, 0.0)
        self.assertGreater(mmae, 1.0)

        mf1, wf1 = calc_macrof1(data)
        self.assertAlmostEqual(mf1, 2.0 / 3.0, places=3)

    def test_cold_start_time_filtering(self):
        stats = [{"end_to_end_latency_ms": 30000.0, "time_to_first_token_ms": 29000.0, "output_token_count": 2, "generation_duration_ms": 1000.0}]
        for _ in range(10):
            stats.append({"end_to_end_latency_ms": 100.0, "time_to_first_token_ms": 20.0, "output_token_count": 2, "generation_duration_ms": 80.0})

        avg_eel, p50_eel, p95_eel, avg_ttft, rps, tps = calc_time_metrics(stats)
        self.assertAlmostEqual(avg_eel, 100.0, places=1)
        self.assertAlmostEqual(p50_eel, 100.0, places=1)
        self.assertAlmostEqual(rps, 10.0, places=1)


class TestPlottingAndSummary(unittest.TestCase):
    def test_pareto_frontier_indices(self):
        # xs (throughput): higher is better; ys (accuracy): higher is better
        xs = [1.0, 5.0, 10.0, 2.0]
        ys = [0.6, 0.7, 0.5, 0.4]
        # (10, 0.5) is highest x; (5, 0.7) has higher y than (10, 0.5); (1, 0.6) has lower y than (5, 0.7).
        # Pareto frontier: index 2 (10, 0.5) and index 1 (5, 0.7).
        p_idx = pareto_frontier_indices(xs, ys, maximize_x=True, maximize_y=True)
        self.assertEqual(set(p_idx), {1, 2})

    def test_get_model_style(self):
        style = get_model_style("gemma4:12b")
        self.assertEqual(style["family"], "Gemma 4")
        self.assertEqual(style["arch"], "Dense")
        self.assertIn("color", style)
        self.assertIn("marker", style)

    def test_load_summary_dataframe(self):
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_summary = {
                "Accuracy": 0.65,
                "Off-by-One Accuracy": 0.92,
                "Macro-F1 Score": 0.64,
                "Weighted-F1 Score": 0.65,
                "Matthews Correlation Coefficient": 0.55,
                "Quadratic Weighted Kappa": 0.78,
                "Macro-Average Mean Absolute Error": 0.45,
                "Mean Absolute Error": 0.42,
                "Mean Squared Error": 0.60,
                "Invalid Prediction Rate": 0.0,
                "Average end-to-end latency": 50.0,
            }
            with open(os.path.join(tmp_dir, "gemma4_12b_sst.json"), "w") as f:
                json.dump(mock_summary, f)
            with open(os.path.join(tmp_dir, "gemma4_12b_semeval.json"), "w") as f:
                json.dump(mock_summary, f)

            df = load_summary_dataframe(tmp_dir)
            self.assertEqual(len(df), 2)
            self.assertIn("Model", df.columns)
            self.assertIn("Accuracy", df.columns)
            self.assertIn("MMAE", df.columns)
            self.assertIn("Throughput", df.columns)
            self.assertIn("Dataset", df.columns)
            self.assertEqual(df.iloc[0]["Model"], "gemma4_12b")
            self.assertEqual(df.iloc[0]["Family"], "Gemma 4")
            self.assertAlmostEqual(df.iloc[0]["Throughput"], 20.0, places=1)

        with tempfile.TemporaryDirectory() as empty_dir:
            df_empty = load_summary_dataframe(empty_dir)
            self.assertTrue(df_empty.empty)

    def test_format_comparison_table(self):
        import pandas as pd
        mock_df = pd.DataFrame([
            {
                "Model": "gemma4_12b",
                "Dataset": "SST-5",
                "Family": "Gemma 4",
                "Arch": "Dense",
                "Params_B": 12.0,
                "Accuracy": 0.65432,
                "Off_By_One": 0.92345,
                "MMAE": 0.45123,
                "QWK": 0.78123,
                "Throughput": 20.456,
                "Latency_ms": 48.9,
                "Invalid_Rate": 0.0,
            }
        ])
        tbl = format_comparison_table(mock_df, dataset="SST-5", top_n=5)
        self.assertEqual(len(tbl), 1)
        self.assertEqual(tbl.iloc[0]["Accuracy"], 0.6543)
        self.assertEqual(tbl.iloc[0]["Throughput"], 20.46)
        self.assertIn("Invalid_Rate_Pct", tbl.columns)

    def test_audit_raw_predictions(self):
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_records = [
                {"actual_label": 2, "expected_label": 2, "raw_model_output": '{"sentiment": 2}'},
                {"actual_label": -1, "expected_label": 1, "raw_model_output": 'I think 1 or 2'},
            ]
            with open(os.path.join(tmp_dir, "test_model_sst.json"), "w") as f:
                json.dump(sample_records, f)

            audit_df = audit_raw_predictions(tmp_dir)
            self.assertEqual(len(audit_df), 1)
            self.assertEqual(audit_df.iloc[0]["Total_Samples"], 2)
            self.assertEqual(audit_df.iloc[0]["Invalid_Count"], 1)
            self.assertEqual(audit_df.iloc[0]["Invalid_Rate_Pct"], 50.0)

    def test_adjust_text_safe_signature(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = ax.annotate("sample", (1, 2))
        # Should not raise exception with or without x, y
        adjust_text_safe([t], ax=ax)
        adjust_text_safe([t], x=[1], y=[2], ax=ax)
        plt.close(fig)

    def test_plot_benchmark_frontier_adaptive_bounds(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        mock_data = pd.DataFrame([
            {"Model": "m1", "Dataset": "SST-5", "Throughput": 5.0, "Accuracy": 0.50, "MMAE": 0.55, "Invalid_Rate": 0.0},
            {"Model": "m2", "Dataset": "SST-5", "Throughput": 10.0, "Accuracy": 0.58, "MMAE": 0.46, "Invalid_Rate": 0.0},
            {"Model": "m3", "Dataset": "SemEval", "Throughput": 4.0, "Accuracy": 0.12, "MMAE": 0.88, "Invalid_Rate": 0.0},
            {"Model": "m4", "Dataset": "SemEval", "Throughput": 8.0, "Accuracy": 0.45, "MMAE": 0.52, "Invalid_Rate": 0.0},
        ])
        fig_acc = plot_benchmark_frontier(mock_data, metric="Accuracy")
        axes_acc = fig_acc.get_axes()
        self.assertEqual(len(axes_acc), 2)
        # SST-5 accuracy [0.50, 0.58] should have bottom > 0.35, not hardcoded 0.15
        sst_acc_ylim = axes_acc[0].get_ylim()
        self.assertGreater(sst_acc_ylim[0], 0.35)
        # SemEval accuracy [0.12, 0.45] should adapt to include 0.12 (below 0.15)
        sem_acc_ylim = axes_acc[1].get_ylim()
        self.assertLessEqual(sem_acc_ylim[0], 0.12)
        plt.close(fig_acc)

        fig_mmae = plot_benchmark_frontier(mock_data, metric="MMAE")
        axes_mmae = fig_mmae.get_axes()
        # MMAE should be inverted (bottom > top) and adaptive
        sst_mmae_ylim = axes_mmae[0].get_ylim()
        self.assertGreater(sst_mmae_ylim[0], sst_mmae_ylim[1])
        # SST-5 MMAE [0.46, 0.55] top should be > 0.0 (adaptive, not hardcoded 0.0)
        self.assertGreater(sst_mmae_ylim[1], 0.1)
        plt.close(fig_mmae)

    def test_plot_benchmark_frontier_excludes_encoders_by_default(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        mock_data = pd.DataFrame([
            {"Model": "llm_1", "Arch": "Dense", "Dataset": "SST-5", "Throughput": 5.0, "Accuracy": 0.50, "MMAE": 0.55, "Invalid_Rate": 0.0},
            {"Model": "llm_2", "Arch": "MoE", "Dataset": "SST-5", "Throughput": 8.0, "Accuracy": 0.52, "MMAE": 0.53, "Invalid_Rate": 0.0},
            {"Model": "enc_1", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 250.0, "Accuracy": 0.45, "MMAE": 0.65, "Invalid_Rate": 0.0},
            {"Model": "llm_1", "Arch": "Dense", "Dataset": "SemEval", "Throughput": 4.0, "Accuracy": 0.50, "MMAE": 0.55, "Invalid_Rate": 0.0},
            {"Model": "enc_1", "Arch": "Encoder", "Dataset": "SemEval", "Throughput": 240.0, "Accuracy": 0.48, "MMAE": 0.62, "Invalid_Rate": 0.0},
        ])
        fig = plot_benchmark_frontier(mock_data, metric="Accuracy")
        axes = fig.get_axes()
        self.assertLess(axes[0].get_xlim()[1], 50.0)
        plt.close(fig)

        fig_inc = plot_benchmark_frontier(mock_data, metric="Accuracy", include_encoders=True)
        axes_inc = fig_inc.get_axes()
        self.assertGreater(axes_inc[0].get_xlim()[1], 200.0)
        plt.close(fig_inc)

    def test_plot_encoder_frontier(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        mock_data = pd.DataFrame([
            {"Model": "gliclass-base-v1.0", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 296.7, "Accuracy": 0.368, "MMAE": 0.82, "Invalid_Rate": 0.0},
            {"Model": "ModernBERT-base-zeroshot-v2.0", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 61.7, "Accuracy": 0.448, "MMAE": 0.71, "Invalid_Rate": 0.0},
            {"Model": "llm_dense", "Arch": "Dense", "Dataset": "SST-5", "Throughput": 5.0, "Accuracy": 0.55, "MMAE": 0.50, "Invalid_Rate": 0.0},
            {"Model": "gliclass-base-v1.0", "Arch": "Encoder", "Dataset": "SemEval", "Throughput": 247.5, "Accuracy": 0.445, "MMAE": 0.67, "Invalid_Rate": 0.0},
            {"Model": "ModernBERT-base-zeroshot-v2.0", "Arch": "Encoder", "Dataset": "SemEval", "Throughput": 45.5, "Accuracy": 0.552, "MMAE": 0.57, "Invalid_Rate": 0.0},
        ])
        fig = plot_encoder_frontier(mock_data, metric="MMAE")
        self.assertIsNotNone(fig)
        axes = fig.get_axes()
        self.assertEqual(len(axes), 2)
        self.assertGreater(axes[0].get_ylim()[0], axes[0].get_ylim()[1])
        plt.close(fig)

        self.assertIsNone(plot_encoder_frontier(pd.DataFrame()))

    def test_plot_architecture_comparison_excludes_encoders_by_default(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        mock_data = pd.DataFrame([
            {"Model": "m1", "Arch": "Dense", "Dataset": "SST-5", "Throughput": 5.0, "Accuracy": 0.50, "MMAE": 0.55, "Invalid_Rate": 0.0},
            {"Model": "m2", "Arch": "MoE", "Dataset": "SST-5", "Throughput": 7.0, "Accuracy": 0.52, "MMAE": 0.52, "Invalid_Rate": 0.0},
            {"Model": "enc1", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 250.0, "Accuracy": 0.45, "MMAE": 0.60, "Invalid_Rate": 0.0},
        ])
        fig = plot_architecture_comparison(mock_data, dataset="SST-5")
        arch_labels = [t.get_text() for t in fig.get_axes()[0].get_xticklabels()]
        self.assertNotIn("Encoder", arch_labels)
        self.assertIn("Dense", arch_labels)
        self.assertIn("MoE", arch_labels)
        plt.close(fig)

        fig_inc = plot_architecture_comparison(mock_data, dataset="SST-5", include_encoders=True)
        arch_labels_inc = [t.get_text() for t in fig_inc.get_axes()[0].get_xticklabels()]
        self.assertIn("Encoder", arch_labels_inc)
        plt.close(fig_inc)

    def test_plot_encoder_comparison(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        mock_data = pd.DataFrame([
            {"Model": "ModernBERT-base-zeroshot-v2.0", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 61.7, "Accuracy": 0.448, "MMAE": 0.71, "Invalid_Rate": 0.0},
            {"Model": "gliclass-base-v1.0", "Arch": "Encoder", "Dataset": "SST-5", "Throughput": 296.7, "Accuracy": 0.368, "MMAE": 0.82, "Invalid_Rate": 0.0},
        ])
        fig = plot_encoder_comparison(mock_data, dataset="SST-5")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.get_axes()), 3)
        plt.close(fig)


class TestLabelerModule(unittest.TestCase):
    def test_format_dataset_input(self):
        import pandas as pd
        row_sst = pd.Series({"text": "Great acting and story"})
        self.assertEqual(format_dataset_input("sst", row_sst), "Great acting and story")

        row_sem = pd.Series({"hashtag": "Oscars", "text": "Loved the speeches"})
        self.assertEqual(format_dataset_input("semeval", row_sem), "Topic: Oscars\nTweet: Loved the speeches")

    def test_skip_existing_outputs(self):
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_sst = os.path.join(tmp_dir, "mock_model_sst.json")
            out_sem = os.path.join(tmp_dir, "mock_model_semeval.json")
            with open(out_sst, "w") as f:
                json.dump([], f)
            with open(out_sem, "w") as f:
                json.dump([], f)

            paths = run_single_model_labeling(
                model_tag="mock:model",
                think=False,
                datasets={"sst": None, "semeval": None},
                dataset_names=["sst", "semeval"],
                overwrite=False,
                output_dir=tmp_dir,
            )
            self.assertEqual(paths["sst"], out_sst)
            self.assertEqual(paths["semeval"], out_sem)

    def test_labeling_record_fields(self):
        import tempfile
        import json
        import pandas as pd

        mock_client = MagicMock()
        mock_client.ps.return_value = {
            "models": [{"name": "test:model", "size": 1000, "size_vram": 1000}]
        }
        mock_client.chat.return_value = [
            {"message": {"content": '{"sentiment": 3}'}, "done": True, "eval_count": 3, "eval_duration": 10, "prompt_eval_duration": 5, "total_duration": 15}
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame([{"text": "A good movie.", "label": 4}])
            paths = run_single_model_labeling(
                model_tag="test:model",
                think=False,
                datasets={"sst": df},
                dataset_names=["sst"],
                overwrite=True,
                output_dir=tmp_dir,
                host="http://custom-host:11434",
                client=mock_client,
            )
            with open(paths["sst"], "r", encoding="utf-8") as f:
                saved = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(saved), 1)
            item = saved[0]
            self.assertEqual(item["expected_label"], 4)
            self.assertEqual(item["actual_label"], 3)
            self.assertNotIn("true_label", item)
            self.assertNotIn("predicted_label", item)
            self.assertFalse(item["is_correct"])
            self.assertEqual(item["ordinal_error"], 1)
            self.assertEqual(item["temperature"], 0.0)
            self.assertEqual(item["seed"], 42)

    def test_process_llm_records_json_and_jsonl(self):
        import tempfile
        import json

        sample_items = [
            {"actual_label": 3, "expected_label": 3, "end_to_end_latency_ms": 50.0},
            {"actual_label": 4, "expected_label": 2, "end_to_end_latency_ms": 60.0},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test JSON array
            json_file = os.path.join(tmp_dir, "model_sst.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(sample_items, f)

            # Test JSON Lines
            jsonl_file = os.path.join(tmp_dir, "model_sst.jsonl")
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for item in sample_items:
                    f.write(json.dumps(item) + "\n")

            res_json = process_llm_records(json_file)
            res_jsonl = process_llm_records(jsonl_file)

            self.assertEqual(res_json["Accuracy"], 0.5)
            self.assertEqual(res_jsonl["Accuracy"], 0.5)
            self.assertEqual(res_json["Average end-to-end latency"], 55.0)
            self.assertEqual(res_jsonl["Average end-to-end latency"], 55.0)


class TestPullModelsModule(unittest.TestCase):
    def test_resolve_model_entries(self):
        main = resolve_model_entries("main")
        self.assertEqual(len(main), 30)

        ablation = resolve_model_entries("ablation")
        self.assertEqual(len(ablation), 8)

        custom = resolve_model_entries("qwen3.5:9b, llama3.1:8b")
        self.assertEqual(len(custom), 2)
        self.assertEqual(custom[0]["tag"], "qwen3.5:9b")
        self.assertEqual(custom[1]["tag"], "llama3.1:8b")

    def test_check_models_status(self):
        targets = [{"tag": "qwen3.5:9b"}, {"tag": "missing:model"}]
        installed = {
            "qwen3.5:9b": {"name": "qwen3.5:9b", "size": 6000000000},
        }
        present, missing = check_models_status(targets, installed)
        self.assertEqual(len(present), 1)
        self.assertEqual(present[0]["tag"], "qwen3.5:9b")
        self.assertEqual(present[0]["installed_size"], 6000000000)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["tag"], "missing:model")

    def test_pull_single_model_mock(self):
        from unittest.mock import patch, MagicMock
        from io import BytesIO
        from src.pull_models import pull_single_model

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = BytesIO(
            b'{"status":"downloading","total":1000,"completed":500,"digest":"abc"}\n'
            b'{"status":"success"}\n'
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            success = pull_single_model("mock_model:latest", "http://127.0.0.1:11434", show_pbar=False)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
