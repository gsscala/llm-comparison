import json
import os
from AUX_FUNCTIONS_TO_GEN_SUM_DATA import *

os.makedirs("../SUMMARIZED_DATA", exist_ok=True)

for model in os.listdir("../new_method"):
    src = os.path.join("../new_method", model)
    try:
        with open(src, "r") as f:
            stats = json.load(f)

        MAE, MSE, OB1, ACC, MMAE = calc_error_metrics(stats)
        MF1 = calc_macrof1(stats)
        QWK = calc_qwk(stats)
        MCC = calc_mcc(stats)
        EEL, TTFT, RPS, TPS = calc_time_metrics(stats)
        data = {
            #"Mean Absolute Error": MAE,
            "Mean Squared Error": MSE,
            "Macro-Average Mean Absolute Error": MMAE,
            #"Off-by-One Accuracy": OB1,
            #"Quadratic Weighted Kappa": QWK,
            #"Matthews Correlation Coefficient": MCC,
            "Macro-F1 Score": MF1,
            "Accuracy": ACC,
            "Average end-to-end latency": EEL,
            #"Average time to first token": TTFT,
            #"Requests per second": RPS,
            #"Output tokens per second": TPS,
        }
    except Exception:
        print(model)
        continue

    dst = os.path.join("../SUMMARIZED_DATA", f"{model}.json")
    with open(dst, "w") as f:
        json.dump(data, f)