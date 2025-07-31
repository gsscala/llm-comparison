import numpy as np


def extract_labels(data):
    """Extract (actual_labels, expected_labels) arrays from a list of instance dicts."""
    actual_labels = []
    expected_labels = []
    for instance in data:
        actual_labels.append(instance.get("actual_label", instance.get("predicted_label", -1)))
        expected_labels.append(instance.get("expected_label", instance.get("true_label", -1)))
    return np.array(actual_labels, dtype=int), np.array(expected_labels, dtype=int)


def calc_error_metrics(data):
    """
    Compute Accuracy, Off-by-One Accuracy, MAE, MSE, and MMAE over all N instances.
    Invalid predictions (< 0 or >= 5) are penalized with the worst-case ordinal error
    max(y_true, 4 - y_true) on the 0..4 scale.
    """
    y_pred, y_true = extract_labels(data)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    valid_mask = (y_pred >= 0) & (y_pred < 5)
    invalid_rate = float(np.mean(~valid_mask))

    accuracy = float(np.mean(valid_mask & (y_pred == y_true)))
    off_by_one = float(np.mean(valid_mask & (np.abs(y_pred - y_true) <= 1)))

    worst_abs_err = np.maximum(y_true, 4 - y_true)
    effective_abs_diff = np.where(valid_mask, np.abs(y_pred - y_true), worst_abs_err)
    effective_sqr_diff = np.where(valid_mask, (y_pred - y_true) ** 2, worst_abs_err ** 2)

    mean_abs_error = float(np.mean(effective_abs_diff))
    mean_sqr_error = float(np.mean(effective_sqr_diff))

    per_class_mmae = []
    valid_per_class_mmae = []
    for c in range(5):
        cls_mask = (y_true == c)
        if np.any(cls_mask):
            per_class_mmae.append(np.mean(effective_abs_diff[cls_mask]))
            valid_cls_mask = cls_mask & valid_mask
            if np.any(valid_cls_mask):
                valid_per_class_mmae.append(np.mean(np.abs(y_pred[valid_cls_mask] - y_true[valid_cls_mask])))

    mmae = float(np.mean(per_class_mmae)) if per_class_mmae else mean_abs_error
    valid_only_mmae = float(np.mean(valid_per_class_mmae)) if valid_per_class_mmae else mmae

    return mean_abs_error, mean_sqr_error, off_by_one, accuracy, mmae, invalid_rate, valid_only_mmae


def calc_qwk(data):
    """
    Compute Quadratic Weighted Kappa over all N instances.
    Invalid predictions (-1) are mapped to the worst-case ordinal label opposite
    to true label (0 if y_true >= 2 else 4) so failures penalize agreement.
    """
    y_pred, y_true = extract_labels(data)
    n = len(y_true)
    if n == 0:
        return 0.0

    valid_mask = (y_pred >= 0) & (y_pred < 5)
    worst_case_label = np.where(y_true >= 2, 0, 4)
    effective_pred = np.where(valid_mask, y_pred, worst_case_label)

    n_ratings = 5
    O = np.zeros((n_ratings, n_ratings), dtype=float)
    np.add.at(O, (y_true, effective_pred), 1.0)

    total = O.sum()
    if total == 0:
        return 0.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / total

    r = np.arange(n_ratings)
    W = (r[:, None] - r[None, :]) ** 2 / float((n_ratings - 1) ** 2)

    observed = (W * O).sum() / total
    expected = (W * E).sum() / total

    if expected == 0.0:
        return 1.0

    qwk = 1.0 - (observed / expected)
    return float(qwk)


def calc_mcc(data):
    """
    Compute multiclass Matthews Correlation Coefficient over a 6x6 matrix
    where index 5 represents invalid/failed predictions (-1).
    """
    y_pred, y_true = extract_labels(data)
    n = len(y_true)
    if n == 0:
        return 0.0

    # Map invalid predictions to class 5
    mapped_pred = np.where((y_pred >= 0) & (y_pred < 5), y_pred, 5)

    n_classes = 6
    C = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(C, (y_true, mapped_pred), 1.0)

    total = C.sum()
    if total == 0:
        return 0.0

    trace = np.trace(C)
    sum_cross = (C * C.T).sum()
    numerator = float(trace * total - sum_cross)

    row_sum = C.sum(axis=1)
    col_sum = C.sum(axis=0)
    sum_row_sq = float((row_sum ** 2).sum())
    sum_col_sq = float((col_sum ** 2).sum())

    denom_term1 = float(total * total - sum_row_sq)
    denom_term2 = float(total * total - sum_col_sq)
    if denom_term1 <= 0.0 or denom_term2 <= 0.0:
        return 0.0

    mcc = numerator / np.sqrt(denom_term1 * denom_term2)
    return float(mcc)


def calc_macrof1(data):
    """
    Compute Macro-F1 and Weighted-F1 across the 5 true sentiment classes (0..4).
    Invalid predictions (-1) are mapped to column 5 (false negatives for the true class).
    """
    y_pred, y_true = extract_labels(data)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    mapped_pred = np.where((y_pred >= 0) & (y_pred < 5), y_pred, 5)

    # 5 true classes x 6 predicted classes
    C = np.zeros((5, 6), dtype=float)
    np.add.at(C, (y_true, mapped_pred), 1.0)

    tp = np.diag(C[:, :5])
    pred_sum = C[:, :5].sum(axis=0)  # column sums for classes 0..4
    true_sum = C.sum(axis=1)         # row sums (all true instances for class c)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * true_sum) / np.sum(true_sum)) if np.sum(true_sum) > 0 else 0.0

    return macro_f1, weighted_f1


def calc_time_metrics(stats):
    """Compute robust latency and throughput statistics, filtering cold-start outliers if present."""
    if not stats:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    e2e = [inst.get("end_to_end_latency_ms", 0.0) for inst in stats]
    ttft = [inst.get("time_to_first_token_ms", 0.0) for inst in stats]
    tpot = [inst.get("time_per_output_token_ms", 0.0) for inst in stats]
    out_tokens = [inst.get("output_token_count", 1) for inst in stats]
    gen_durations = [inst.get("generation_duration_ms", 0.0) for inst in stats]

    e2e_arr = np.array(e2e, dtype=float)
    ttft_arr = np.array(ttft, dtype=float)

    # Filter row 0 if it was a cold-start outlier (e.g. 30s model load in legacy files)
    if len(e2e_arr) > 5:
        tail_median = np.median(e2e_arr[1:])
        if tail_median > 0 and e2e_arr[0] > 5.0 * tail_median:
            e2e_arr = e2e_arr[1:]
            ttft_arr = ttft_arr[1:]

    avg_eel = float(np.mean(e2e_arr))
    p50_eel = float(np.median(e2e_arr))
    p95_eel = float(np.percentile(e2e_arr, 95))
    avg_ttft = float(np.mean(ttft_arr))
    rps = float(1000.0 / avg_eel) if avg_eel > 0 else 0.0

    total_tokens = sum(out_tokens)
    total_gen_ms = sum(gen_durations)
    if total_gen_ms > 0:
        tps = float(total_tokens / (total_gen_ms / 1000.0))
    else:
        valid_tpot = [t for t in tpot if t > 0]
        tps = float(1000.0 / np.mean(valid_tpot)) if valid_tpot else 0.0

    return avg_eel, p50_eel, p95_eel, avg_ttft, rps, tps