"""Check model availability on Ollama server and pull missing checkpoints."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PRINT_LOCK = threading.Lock()


def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import MAIN_LLM_MODELS, THINKING_ABLATION_MODELS


def get_installed_models(host: str) -> Dict[str, dict]:
    """Query the Ollama server for all currently installed models."""
    url = f"{host.rstrip('/')}/api/tags"
    req = urllib.request.Request(url, headers={"User-Agent": "llm-comparison-checker"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise ConnectionError(
            f"Could not connect to Ollama server at {host}: {exc}\n"
            f"Make sure the SSH tunnel is active (e.g. ssh -J ... -L 11434:localhost:11434)."
        )

    installed = {}
    for item in data.get("models", []):
        name = item.get("name", "")
        model = item.get("model", "")
        if name:
            installed[name] = item
        if model:
            installed[model] = item
        # Also map without :latest if present
        if name.endswith(":latest"):
            installed[name[:-7]] = item
    return installed


def check_models_status(
    target_entries: List[dict],
    installed_models: Dict[str, dict],
) -> Tuple[List[dict], List[dict]]:
    """Partition target models into present and missing lists."""
    present = []
    missing = []

    for entry in target_entries:
        tag = entry["tag"]
        normalized_tag = tag if ":" in tag else f"{tag}:latest"
        base_tag = tag[:-7] if tag.endswith(":latest") else tag

        match = None
        if tag in installed_models:
            match = installed_models[tag]
        elif normalized_tag in installed_models:
            match = installed_models[normalized_tag]
        elif base_tag in installed_models:
            match = installed_models[base_tag]

        if match is not None:
            entry_copy = dict(entry)
            entry_copy["installed_size"] = match.get("size", 0)
            present.append(entry_copy)
        else:
            missing.append(entry)

    return present, missing


def pull_single_model(tag: str, host: str, show_pbar: bool = True) -> bool:
    """Stream download for a single model from the Ollama registry."""
    url = f"{host.rstrip('/')}/api/pull"
    payload = json.dumps({"name": tag, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "llm-comparison-puller"},
    )

    if show_pbar:
        print(f"\nPulling {tag}...")
    else:
        safe_print(f"  [START] {tag}")

    pbar = None
    current_digest = None
    last_reported_pct = -1

    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                if "error" in event:
                    if pbar:
                        pbar.close()
                    safe_print(f"Error pulling {tag}: {event['error']}", file=sys.stderr)
                    return False

                status = event.get("status", "")
                total = event.get("total", 0)
                completed = event.get("completed", 0)
                digest = event.get("digest", "")

                if show_pbar:
                    if total > 0 and tqdm is not None:
                        if digest != current_digest:
                            if pbar:
                                pbar.close()
                            current_digest = digest
                            short_digest = digest[:12] if digest else "layer"
                            pbar = tqdm(
                                total=total,
                                initial=completed,
                                unit="B",
                                unit_scale=True,
                                desc=f"[{short_digest}]",
                                leave=False,
                            )
                        elif pbar:
                            pbar.n = completed
                            pbar.refresh()
                    elif status and not status.startswith("downloading"):
                        print(f"  {status}")
                else:
                    if total > 0:
                        pct = int(completed * 100 / total)
                        if pct >= last_reported_pct + 25 or pct == 100:
                            last_reported_pct = pct
                            comp_gb = round(completed / 1e9, 2)
                            tot_gb = round(total / 1e9, 2)
                            safe_print(f"  [{tag}] {pct}% ({comp_gb} / {tot_gb} GB)")
                    elif status and not status.startswith("downloading"):
                        safe_print(f"  [{tag}] {status}")

        if pbar:
            pbar.close()
        safe_print(f"  [DONE]  {tag} pulled successfully.")
        return True

    except Exception as exc:
        if pbar:
            pbar.close()
        safe_print(f"Network error pulling {tag}: {exc}", file=sys.stderr)
        return False


def resolve_model_entries(preset_or_tags: str) -> List[dict]:
    """Resolve CLI string into model entries list."""
    preset = preset_or_tags.strip().lower()
    if preset == "main":
        return list(MAIN_LLM_MODELS)
    if preset == "ablation":
        return list(THINKING_ABLATION_MODELS)
    if preset == "all":
        seen = set()
        merged = []
        for m in MAIN_LLM_MODELS + THINKING_ABLATION_MODELS:
            if m["tag"] not in seen:
                seen.add(m["tag"])
                merged.append(m)
        return merged

    tags = [t.strip() for t in preset_or_tags.split(",") if t.strip()]
    return [{"tag": t} for t in tags]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model availability on the Ollama server and pull missing checkpoints."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="main",
        help="Model preset ('main', 'ablation', 'all') or comma-separated tags.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama host URL (defaults to OLLAMA_HOST or http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Proceed to pull missing models without an interactive confirmation prompt.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only display availability status without pulling any models.",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent model downloads (default: 3). Set to 1 for sequential with progress bar.",
    )

    args = parser.parse_args()
    host = args.host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    print(f"Connecting to Ollama server at {host}...")
    try:
        installed = get_installed_models(host)
    except ConnectionError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    targets = resolve_model_entries(args.models)
    present, missing = check_models_status(targets, installed)

    print("\n" + "=" * 60)
    print(f"Model status check: {len(present)} available, {len(missing)} missing")
    print("=" * 60)

    if present:
        print("\nAvailable models on server:")
        for m in present:
            size_gb = round(m.get("installed_size", 0) / 1e9, 2)
            print(f"  [OK]      {m['tag']:<28} ({size_gb} GB)")

    if missing:
        print("\nMissing models:")
        for m in missing:
            est = m.get("ollama_size_gb")
            est_str = f"~{est} GB" if est else "size unknown"
            lab = m.get("lab", "")
            meta = f"[{lab}] {est_str}" if lab else est_str
            print(f"  [MISSING] {m['tag']:<28} ({meta})")

    if not missing:
        print("\nAll requested models are already available on the server. Ready to run.")
        sys.exit(0)

    if args.check_only:
        print(f"\nCheck complete. {len(missing)} models need to be pulled before running full benchmark.")
        sys.exit(0)

    # Prompt user unless -y / --yes was supplied
    if not args.yes:
        print()
        prompt_text = f"Do you want to pull {len(missing)} missing models to the remote server? [y/N]: "
        try:
            choice = input(prompt_text).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)

        if choice not in ("y", "yes"):
            print("Aborted. No models were pulled.")
            sys.exit(0)

    # Execute pulls
    concurrency = max(1, args.concurrency)
    print(f"\nStarting pull sequence for {len(missing)} models (concurrency: {concurrency})...")
    success_count = 0
    fail_count = 0

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_model = {
                executor.submit(pull_single_model, m["tag"], host, False): m["tag"]
                for m in missing
            }
            for future in as_completed(future_to_model):
                tag = future_to_model[future]
                try:
                    ok = future.result()
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as exc:
                    safe_print(f"Exception pulling {tag}: {exc}", file=sys.stderr)
                    fail_count += 1
    else:
        for i, m in enumerate(missing, start=1):
            tag = m["tag"]
            print(f"\n[{i}/{len(missing)}] Processing {tag}")
            ok = pull_single_model(tag, host, show_pbar=True)
            if ok:
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "=" * 60)
    print(f"Pull sequence complete: {success_count} succeeded, {fail_count} failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
