import json
import time
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import pipeline
from imblearn.metrics import macro_averaged_mean_absolute_error
from sklearn.metrics import accuracy_score
from transformers.pipelines.pt_utils import KeyDataset

semeval = pd.read_csv(
    "semeval.txt",
    sep='\t',
    header=None,
    names=['tweet_id', 'hashtag', 'label', 'text'],
    encoding='utf-8',
    quoting=3
)

sst = pd.DataFrame(load_dataset("SetFit/sst5")["test"])

semeval['label'] = semeval['label'].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
semeval = semeval.drop(columns=['tweet_id'])
sst = sst.drop(columns=['label_text'])

MODELS_TO_EVALUATE = {
    "BART": "facebook/bart-large-mnli",
    "DeBERTa-v3": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "ModernBERT": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
    "RoBERTa": "roberta-large-mnli" 
}

CANDIDATE_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]

LABEL_TO_INT = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4
}

# Assuming 'sst' and 'semeval' are predefined pandas DataFrames
datasets_to_process = {
    "sst": sst,
    "semeval": semeval
}

for dataset_name, df in datasets_to_process.items():
    print(f"\n{'='*50}\nEvaluating Dataset: {dataset_name}\n{'='*50}")
    
    # 1. Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)

    # 2. Standardize SemEval Columns and Combine Context
    if dataset_name == "semeval":
        def combine_text_context(example):
            example["text"] = f"Context: {example['hashtag']} | Text: {example['text']}"
            return example
        
        hf_dataset = hf_dataset.map(combine_text_context)

    # 3. Extract texts and true labels for evaluation
    texts = hf_dataset["text"]
    true_labels = hf_dataset["label"]

    for model_label, model_id in MODELS_TO_EVALUATE.items():
        print(f"\n--- Running Zero-Shot Inference with {model_label} ---")
        
        # Initialize Pipeline
        classifier = pipeline(
            "zero-shot-classification", 
            model=model_id, 
            device=0 # Uses GPU
        )
        
        print(f"Classifying {len(texts)} examples...")
        
# --- START TIMING ---
        start_time = time.perf_counter()
        
        # Use KeyDataset to stream batches properly
        predictions = classifier(
            KeyDataset(hf_dataset, "text"), 
            candidate_labels=CANDIDATE_LABELS,
            batch_size=1 
        )
        
        # Process Predictions (This will now correctly loop 2,210 times)
        predicted_ints = []
        for pred in predictions:
            top_label = pred["labels"][0] 
            predicted_ints.append(LABEL_TO_INT[top_label])
   
        # --- END TIMING ---
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time


        # Cast true labels to standard Python int to prevent JSON serialization errors
        clean_true_labels = [int(lbl) for lbl in true_labels]

        results = {
            "Accuracy" : accuracy_score(clean_true_labels, predicted_ints),
            "Macro-Average Mean Absolute Error" : macro_averaged_mean_absolute_error(clean_true_labels, predicted_ints),
            "Average end-to-end latency" : elapsed_time / len(texts),
            "prediction/true" : list(zip(predicted_ints, clean_true_labels))
        }
  
        # Added .json extension
        file_name = f"{model_label}_{dataset_name}.json"
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Saved results to {file_name}")

        # 5. Prevent GPU Memory Leaks
        del classifier
        torch.cuda.empty_cache()
