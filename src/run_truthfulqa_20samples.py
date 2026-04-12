"""
TruthfulQA inference runner — first 20 samples.

Usage:
    python src/run_truthfulqa_20samples.py
    python src/run_truthfulqa_20samples.py --data_file data/TruthfulQA.json --limit 20

Output:
    result/TRUTHFULQA/allenai_olmo-3.1-32b-instruct/truthfulqa_20samples.json
"""

from __future__ import annotations

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import requests

# ─── Constants ────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "allenai/olmo-3.1-32b-instruct"
MODEL_ORDER = 42
TEMPERATURE = 0.0
MAX_TOKENS = 512

OUTPUT_DIR = Path("result/TRUTHFULQA/allenai_olmo-3.1-32b-instruct")
OUTPUT_FILE = OUTPUT_DIR / "truthfulqa_20samples.json"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_file: str, limit: int = 20) -> list[dict]:
    """Load TruthfulQA JSON and return first `limit` samples."""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"{data_file} is empty or not a list.")

    samples = data[:limit]

    # Debug: print keys of first sample
    print(f"[DEBUG] Dataset keys: {list(samples[0].keys())}")
    print(f"[INFO]  Loaded {len(samples)} samples from {data_file}")
    return samples


def load_model_config(model_order: int = MODEL_ORDER) -> dict:
    """Load model config entry by order from config/model_keys.json."""
    with open("config/model_keys.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    for m in models:
        if m.get("order") == model_order:
            return m

    raise RuntimeError(f"Model order {model_order} not found in config/model_keys.json")


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    """Build strict TruthfulQA prompt — no extra instructions."""
    return f"Q: {question}\nA:"


# ─── Model call ───────────────────────────────────────────────────────────────

def run_model(
    model_cfg: dict,
    prompt: str,
    retry_count: int = 0,
    max_retries: int = 3,
) -> dict:
    """Call OpenRouter API with exponential-backoff retry."""
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key_alias']}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    start = time.time()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start

        if resp.status_code == 200:
            api_resp = resp.json()
            content = api_resp["choices"][0]["message"].get("content", "").strip()
            return {
                "success": True,
                "model_answer": content,
                "latency": latency,
                "usage": api_resp.get("usage", {}),
                "error": None,
                "retry_count": retry_count,
            }

        if resp.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            wait = 3 * (2 ** retry_count)
            print(f"      [RETRY] HTTP {resp.status_code} — waiting {wait}s …")
            time.sleep(wait)
            return run_model(model_cfg, prompt, retry_count + 1, max_retries)

        return {
            "success": False,
            "model_answer": None,
            "latency": latency,
            "usage": {},
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "retry_count": retry_count,
        }

    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait = 5 * (2 ** retry_count)
            print(f"      [RETRY] Timeout — waiting {wait}s …")
            time.sleep(wait)
            return run_model(model_cfg, prompt, retry_count + 1, max_retries)
        return {
            "success": False,
            "model_answer": None,
            "latency": time.time() - start,
            "usage": {},
            "error": "Timeout after retries",
            "retry_count": retry_count,
        }

    except Exception as exc:
        if retry_count < max_retries:
            wait = 3 * (2 ** retry_count)
            print(f"      [RETRY] Exception — waiting {wait}s …")
            time.sleep(wait)
            return run_model(model_cfg, prompt, retry_count + 1, max_retries)
        return {
            "success": False,
            "model_answer": None,
            "latency": time.time() - start,
            "usage": {},
            "error": str(exc),
            "retry_count": retry_count,
        }


# ─── Output ───────────────────────────────────────────────────────────────────

def save_output(results: list[dict], output_path: Path) -> None:
    """Write results list to JSON, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TruthfulQA 20-sample inference runner")
    parser.add_argument("--data_file", default="data/TruthfulQA.json", help="Path to TruthfulQA JSON")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to run (default: 20)")
    parser.add_argument("--model_order", type=int, default=MODEL_ORDER, help="Model order in config/model_keys.json")
    args = parser.parse_args()

    print("=" * 60)
    print("TRUTHFULQA INFERENCE RUNNER")
    print("=" * 60)

    model_cfg = load_model_config(args.model_order)
    print(f"[INFO]  Model : {model_cfg['name']} ({model_cfg['model_id']})")
    print(f"[INFO]  Temp  : {TEMPERATURE}")

    samples = load_data(args.data_file, args.limit)
    print(f"[INFO]  Output: {OUTPUT_FILE.resolve()}")
    print("=" * 60)

    results: list[dict] = []
    start_all = time.time()

    for idx, sample in enumerate(samples, start=1):
        question        = sample.get("question", "")
        best_answer     = sample.get("answer", "")          # TruthfulQA uses "answer" as the best answer
        correct_answers = sample.get("correct_answers") or []
        incorrect_answers = sample.get("incorrect_answers") or []

        prompt = build_prompt(question)
        api_result = run_model(model_cfg, prompt)

        model_answer = api_result["model_answer"] or ""

        # ── Debug print ──────────────────────────────────────────
        print(f"\n[{idx:02d}/{len(samples)}] Q: {question}")
        print(f"        A: {model_answer}")
        status = "OK" if api_result["success"] else "FAIL"
        print(f"        [{status}] latency={api_result['latency']:.2f}s"
              + (f"  error={api_result['error']}" if not api_result["success"] else ""))

        results.append({
            "index": idx,
            "question": question,
            "model_answer": model_answer,
            "best_answer": best_answer,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "metadata": {
                "model_id": model_cfg["model_id"],
                "temperature": TEMPERATURE,
                "latency_sec": round(api_result["latency"], 4),
                "usage": api_result["usage"],
                "success": api_result["success"],
                "error": api_result["error"],
                "retry_count": api_result["retry_count"],
                "timestamp": datetime.now().isoformat(),
            },
        })

        # Rate-limit courtesy delay between requests
        if idx < len(samples):
            time.sleep(0.5)

    elapsed = time.time() - start_all

    save_output(results, OUTPUT_FILE)

    # ── Final summary ────────────────────────────────────────────
    success_count = sum(1 for r in results if r["metadata"]["success"])
    print("\n" + "=" * 60)
    print(f"DONE  {success_count}/{len(results)} successful  |  {elapsed:.1f}s total")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print("=" * 60)

    # Confirm file exists
    if OUTPUT_FILE.exists():
        print(f"[OK] File confirmed: {OUTPUT_FILE.resolve()}")
    else:
        print("[ERROR] Output file was NOT created.")


if __name__ == "__main__":
    main()
