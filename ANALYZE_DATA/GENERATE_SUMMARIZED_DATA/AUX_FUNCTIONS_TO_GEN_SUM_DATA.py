import numpy as np
# import pygmo as pg

def calc_error_metrics(data: dict):
    predicted_labels = []
    true_labels = []
    for instance in data:
        predicted_labels.append(instance["predicted_label"])
        true_labels.append(instance["true_label"])
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
    mask = (predicted_labels >= 0) & (predicted_labels < 5) & (true_labels >= 0) & (true_labels < 5)
    if not np.any(mask):
        return 0.0
    predicted_labels = predicted_labels[mask]
    true_labels = true_labels[mask]
    
    diff = predicted_labels - true_labels
    sqr_diff = diff * diff
    abs_diff = np.abs(diff)
    mean_sqr_error = np.mean(sqr_diff)
    mean_abs_error = np.mean(abs_diff)
    off_by_one = np.mean(abs_diff <= 1)
    
    return float(mean_abs_error), float(mean_sqr_error), float(off_by_one)

def calc_qwk(data: dict):
    predicted_labels = []
    true_labels = []
    for instance in data:
        predicted_labels.append(instance["predicted_label"])
        true_labels.append(instance["true_label"])
    if len(predicted_labels) == 0:
        return 0.0

    y_pred = np.array(predicted_labels, dtype=int)
    y_true = np.array(true_labels, dtype=int)

    # keep only valid 0..4 labels
    mask = (y_pred >= 0) & (y_pred < 5) & (y_true >= 0) & (y_true < 5)
    if not np.any(mask):
        return 0.0
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    n_ratings = 5
    O = np.zeros((n_ratings, n_ratings), dtype=float)
    np.add.at(O, (y_true, y_pred), 1.0)

    N = O.sum()
    if N == 0:
        return 0.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N

    r = np.arange(n_ratings)
    W = (r[:, None] - r[None, :]) ** 2 / float((n_ratings - 1) ** 2)

    observed = (W * O).sum() / N
    expected = (W * E).sum() / N

    if expected == 0.0:
        return 1.0

    qwk = 1.0 - (observed / expected)
    return float(qwk)

def calc_mcc(data: dict):
    predicted_labels = []
    true_labels = []
    for instance in data:
        predicted_labels.append(instance["predicted_label"])
        true_labels.append(instance["true_label"])
    if len(predicted_labels) == 0:
        return 0.0

    y_pred = np.array(predicted_labels, dtype=int)
    y_true = np.array(true_labels, dtype=int)

    # keep only valid 0..4 labels
    mask = (y_pred >= 0) & (y_pred < 5) & (y_true >= 0) & (y_true < 5)
    if not np.any(mask):
        return 0.0
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    n_classes = 5
    C = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(C, (y_true, y_pred), 1.0)

    N = C.sum()
    if N == 0:
        return 0.0

    trace = np.trace(C)
    sum_cross = (C * C.T).sum()
    numerator = float(trace * trace - sum_cross)

    row_sum = C.sum(axis=1)
    col_sum = C.sum(axis=0)
    sum_row_sq = float((row_sum ** 2).sum())
    sum_col_sq = float((col_sum ** 2).sum())

    denom_term1 = float(N * N - sum_row_sq)
    denom_term2 = float(N * N - sum_col_sq)
    if denom_term1 <= 0.0 or denom_term2 <= 0.0:
        return 0.0

    mcc = numerator / np.sqrt(denom_term1 * denom_term2)
    return float(mcc)

def calc_macrof1(data: dict):
    predicted_labels = []
    true_labels = []
    for instance in data:
        predicted_labels.append(instance["predicted_label"])
        true_labels.append(instance["true_label"])
    if len(predicted_labels) == 0:
        return 0.0

    y_pred = np.array(predicted_labels, dtype=int)
    y_true = np.array(true_labels, dtype=int)

    # keep only valid 0..4 labels (robustness)
    mask = (y_pred >= 0) & (y_pred < 5) & (y_true >= 0) & (y_true < 5)
    if not np.any(mask):
        return 0.0
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    n_classes = 5
    C = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(C, (y_true, y_pred), 1.0)

    tp = np.diag(C)
    pred_sum = C.sum(axis=0)
    true_sum = C.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    return float(np.mean(f1))

def calc_time_metrics(stats:dict):
    end_to_end_latency_ms = []
    time_to_first_token_ms = []
    time_per_output_token_ms = []
    
    for instance in stats:
        end_to_end_latency_ms.append(instance["end_to_end_latency_ms"])
        time_to_first_token_ms.append(instance["time_to_first_token_ms"])
        time_per_output_token_ms.append(instance["time_per_output_token_ms"])
    
    end_to_end_latency_ms = np.array(end_to_end_latency_ms)
    time_to_first_token_ms = np.array(time_to_first_token_ms)
    
    avg_eel = np.mean(end_to_end_latency_ms)
    avg_ttft = np.mean(time_to_first_token_ms)
    avg_tpot = np.mean(time_per_output_token_ms)
    rps = 1 / avg_eel
    tps = 1000 / avg_tpot
    
    return avg_eel, avg_ttft, rps, tps

# def find_best_model_by_hv_contribution(models):
#     """
#     Finds the best model in a set of non-dominated solutions using the
#     hypervolume contribution method for a maximization problem.

#     Args:
#         models (np.ndarray): A 2D numpy array where each row is a model
#                              and each column is an objective value.

#     Returns:
#         tuple: A tuple containing the best model (np.ndarray) and its
#                hypervolume contribution (float).
#     """
#     # 1. Define the Reference Point (for maximization)
#     # The reference point must be worse than any model in all objectives.
#     # We find the minimum value for each objective and subtract a small amount.
#     ref_point = np.min(models, axis=0) - 1
#     print(f"Calculated Reference Point: {ref_point}\n")

#     # 2. Calculate the Total Hypervolume
#     # pygmo's hypervolume calculator assumes minimization, so we must invert
#     # our models and reference point to simulate a maximization problem.
#     hv = pg.hypervolume(-models)
#     total_hv = hv.compute(-ref_point)
#     print(f"Total Hypervolume of the set: {total_hv:.4f}\n")

#     # 3. Calculate Each Model's Contribution
#     contributions = []
#     for i in range(len(models)):
#         # Create a temporary set without the current model
#         temp_models = np.delete(models, i, axis=0)

#         # Calculate the hypervolume of the temporary set
#         temp_hv_calculator = pg.hypervolume(-temp_models)
#         hv_without_model = temp_hv_calculator.compute(-ref_point)

#         # The contribution is the difference
#         contribution = total_hv - hv_without_model
#         contributions.append(contribution)
#         print(f"Model {i+1} {models[i]}: Contribution = {contribution:.4f}")

#     # 4. Identify the Best Model
#     best_model_index = np.argmax(contributions)
#     best_model = models[best_model_index]
#     max_contribution = contributions[best_model_index]

#     return best_model, max_contribution