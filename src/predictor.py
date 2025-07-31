import json
import os
import re
import time
try:
    import ollama
except ImportError:
    ollama = None

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
        }
    },
    "required": ["sentiment"],
    "additionalProperties": False,
}


def parse_sentiment_to_id(text: str) -> int:
    """Parse a 0..4 sentiment label from structured JSON or free text, returning -1 on failure."""
    if not text or not isinstance(text, str):
        return -1

    cleaned = text.strip()

    # If a <think> block is present, only parse text after </think>
    if "<think>" in cleaned:
        if "</think>" not in cleaned:
            return -1
        cleaned = cleaned.split("</think>", 1)[1].strip()

    # 1. Try direct JSON parsing first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "sentiment" in obj:
            val = int(obj["sentiment"])
            return val if 0 <= val <= 4 else -1
    except Exception:
        pass

    # 2. Try extracting an embedded JSON object {"sentiment": <int>}
    json_match = re.search(r'\{\s*"sentiment"\s*:\s*(-?\d+)\s*\}', cleaned)
    if json_match:
        val = int(json_match.group(1))
        return val if 0 <= val <= 4 else -1

    # 3. Fallback: check for standalone digits in 0..4; reject ambiguous multi-label rambles
    matches = re.findall(r"\b[0-4]\b", cleaned)
    unique_matches = set(matches)
    if len(unique_matches) == 1:
        return int(matches[0])

    return -1


class Predictor:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        think: bool = False,
        num_ctx: int | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        host: str | None = None,
        warmup: bool = True,
        structured_output: bool | None = None,
        client: object | None = None,
    ):
        self.model = model_name
        self.prompt = prompt_template
        self.think = think
        self.num_ctx = num_ctx or (4096 if think else 1024)
        self.temperature = 0.0
        self.seed = int(seed)
        self.host = host or os.environ.get("OLLAMA_HOST")
        self.structured_output = (not think) if structured_output is None else structured_output
        if client is not None:
            self.client = client
        elif ollama is not None:
            self.client = ollama.Client(host=self.host)
        else:
            raise ImportError(
                "The 'ollama' package is required for running Ollama models. "
                "Install it with: pip install ollama"
            )
        self.vram_used_mib = None
        self.gpu_residency_pct = None
        if warmup:
            self._warmup()

    def _chat_kwargs(self, full_prompt: str, stream: bool = True, num_predict: int | None = None) -> dict:
        max_tokens = num_predict or (2048 if self.think else 32)
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
            "keep_alive": "15m",
        }
        if self.structured_output:
            kwargs["format"] = SENTIMENT_SCHEMA
        # Pass think flag when supported by the ollama client / server
        kwargs["think"] = self.think
        return kwargs

    def _call_chat(self, **kwargs):
        """Invoke client.chat and gracefully retry without `think` if an older model rejects the flag."""
        try:
            return self.client.chat(**kwargs)
        except Exception as exc:
            if "think" in kwargs and ("think" in str(exc).lower() or "unexpected keyword" in str(exc).lower()):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("think", None)
                return self.client.chat(**fallback_kwargs)
            raise

    def _warmup(self):
        """Pre-load the model into GPU VRAM and record residency percentage from ollama.ps()."""
        warmup_prompt = self._format_prompt("Great movie, really enjoyed the acting.")
        kwargs = self._chat_kwargs(warmup_prompt, stream=False, num_predict=16)
        self._call_chat(**kwargs)

        try:
            ps_info = self.client.ps()
            models_list = ps_info.get("models", []) if isinstance(ps_info, dict) else getattr(ps_info, "models", [])
            for m in models_list:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", "")
                model_tag = m.get("model") if isinstance(m, dict) else getattr(m, "model", "")
                if self.model in (name, model_tag) or name.startswith(self.model):
                    size = (m.get("size") if isinstance(m, dict) else getattr(m, "size", 0)) or 0
                    size_vram = (m.get("size_vram") if isinstance(m, dict) else getattr(m, "size_vram", 0)) or 0
                    self.vram_used_mib = round(size_vram / (1024 * 1024), 1)
                    self.gpu_residency_pct = round(100.0 * size_vram / size, 1) if size else 0.0
                    if size_vram < size:
                        print(
                            f"NOTE: {self.model} partially offloaded to system RAM: "
                            f"VRAM={self.vram_used_mib} MiB ({self.gpu_residency_pct}% on GPU)"
                        )
                    break
        except Exception:
            pass

    def unload(self):
        """Evict the model from VRAM immediately before loading the next model."""
        try:
            self.client.chat(model=self.model, messages=[], keep_alive=0)
        except Exception:
            pass

    def _format_prompt(self, input_text: str) -> str:
        if "{sentence}" in self.prompt:
            return self.prompt.replace("{sentence}", input_text)
        return f"{self.prompt}\n{input_text}"

    def predict(self, input_text: str) -> dict:
        raw_output_chunks = []
        thinking_chunks = []
        first_token_received_time = None
        final_chunk = {}

        full_prompt = self._format_prompt(input_text)
        prediction_record = {
            "prompt": self.prompt,
            "instance": input_text,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "seed": self.seed,
            "vram_used_mib": self.vram_used_mib,
            "gpu_residency_pct": self.gpu_residency_pct,
        }

        start_time = time.perf_counter()
        stream = self._call_chat(**self._chat_kwargs(full_prompt, stream=True))

        for chunk in stream:
            msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
            content_piece = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            thinking_piece = msg.get("thinking", "") if isinstance(msg, dict) else getattr(msg, "thinking", "")

            if (content_piece or thinking_piece) and first_token_received_time is None:
                first_token_received_time = time.perf_counter()

            if thinking_piece:
                thinking_chunks.append(thinking_piece)
            if content_piece:
                raw_output_chunks.append(content_piece)

            is_done = chunk.get("done", False) if isinstance(chunk, dict) else getattr(chunk, "done", False)
            if is_done:
                final_chunk = chunk

        end_time = time.perf_counter()

        wall_e2e_ms = (end_time - start_time) * 1000.0
        raw_content = "".join(raw_output_chunks)
        thinking_text = "".join(thinking_chunks)

        if thinking_text:
            prediction_record["thinking_trace"] = thinking_text

        prediction_record["raw_model_output"] = raw_content

        # Extract native token counts and durations from Ollama's final chunk when available
        eval_count = (
            final_chunk.get("eval_count")
            if isinstance(final_chunk, dict)
            else getattr(final_chunk, "eval_count", None)
        )
        eval_duration_ns = (
            final_chunk.get("eval_duration")
            if isinstance(final_chunk, dict)
            else getattr(final_chunk, "eval_duration", None)
        )
        prompt_eval_duration_ns = (
            final_chunk.get("prompt_eval_duration")
            if isinstance(final_chunk, dict)
            else getattr(final_chunk, "prompt_eval_duration", None)
        )
        total_duration_ns = (
            final_chunk.get("total_duration")
            if isinstance(final_chunk, dict)
            else getattr(final_chunk, "total_duration", None)
        )

        prediction_record["end_to_end_latency_ms"] = (
            total_duration_ns / 1e6 if total_duration_ns else wall_e2e_ms
        )
        prediction_record["output_token_count"] = (
            int(eval_count) if eval_count is not None else len(raw_output_chunks)
        )

        if first_token_received_time is not None:
            wall_ttft_ms = (first_token_received_time - start_time) * 1000.0
        else:
            wall_ttft_ms = prediction_record["end_to_end_latency_ms"]

        prediction_record["time_to_first_token_ms"] = (
            prompt_eval_duration_ns / 1e6 if prompt_eval_duration_ns else wall_ttft_ms
        )

        if eval_duration_ns and prediction_record["output_token_count"] > 0:
            gen_ms = eval_duration_ns / 1e6
            prediction_record["generation_duration_ms"] = gen_ms
            prediction_record["time_per_output_token_ms"] = gen_ms / prediction_record["output_token_count"]
        else:
            gen_ms = max(0.0, wall_e2e_ms - wall_ttft_ms)
            prediction_record["generation_duration_ms"] = gen_ms
            if prediction_record["output_token_count"] > 1:
                prediction_record["time_per_output_token_ms"] = gen_ms / (prediction_record["output_token_count"] - 1)
            else:
                prediction_record["time_per_output_token_ms"] = gen_ms
        pred = parse_sentiment_to_id(raw_content)
        prediction_record["actual_label"] = pred
        return prediction_record