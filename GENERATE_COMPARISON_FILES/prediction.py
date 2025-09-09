import ollama
import time
import re

class Predictor:
    def __init__(self, model_name, prompt_template):
        self.model = model_name
        self.prompt = prompt_template
        
    def predict(self, input_text):
        raw_output_chunks = []
        first_token_received_time = None
        start_time = time.perf_counter()

        full_prompt = f"{self.prompt}\n{input_text}"
        
        prediction_record = {
            'prompt': self.prompt,
            'instance': input_text
        }
        
        stream = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=True,
            options={'temperature': 0}
        )

        for chunk in stream:
            if first_token_received_time is None:
                first_token_received_time = time.perf_counter()
            raw_output_chunks.append(chunk['message']['content'])

        end_time = time.perf_counter()

        # 4. Populate the record with collected data
        prediction_record['end_to_end_latency_ms'] = (end_time - start_time) * 1000
        prediction_record['raw_model_output'] = "".join(raw_output_chunks)
        prediction_record['output_token_count'] = len(raw_output_chunks) # Approximation, better tokenizers exist

        if first_token_received_time:
            prediction_record['time_to_first_token_ms'] = (first_token_received_time - start_time) * 1000
            generation_time_ms = (end_time - first_token_received_time) * 1000
            if prediction_record['output_token_count'] > 1:
                prediction_record['time_per_output_token_ms'] = generation_time_ms / (prediction_record['output_token_count'] - 1)
            else:
                prediction_record['time_per_output_token_ms'] = generation_time_ms
        else: # Handles cases where there is no output
            prediction_record['time_to_first_token_ms'] = prediction_record['end_to_end_latency_ms']
            prediction_record['time_per_output_token_ms'] = 0

        def parse_sentiment_to_id(text):
            matches = re.findall(r'\d+', text)
            return int(matches[-1]) if matches else -1
            

        prediction_record['predicted_label'] = parse_sentiment_to_id(prediction_record['raw_model_output'])

        return prediction_record