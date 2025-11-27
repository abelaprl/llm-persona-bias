from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

import sys
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
from load_keys import load_model_keys, get_models, find_model_by_name


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def ensure_dir(path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def load_gsm8k(path: str = "data/gsm8k.json") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GSM8K file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of objects with at least: {"id", "question", "answer"}
    if not isinstance(data, list):
        raise ValueError("GSM8K JSON must be a list of question objects")
    return data


def load_implicit_persona(path: str, persona_id: int) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Implicit persona file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        raise ValueError("Implicit persona JSON must be a list of objects {id, prompt}")
    for obj in arr:
        if obj.get("id") == persona_id:
            return obj.get("prompt")
    return None


def call_openrouter(model_id: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_text(response_json: Dict[str, Any]) -> str:
    try:
        choices = response_json.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
    except Exception:
        pass
    return ""


def main():
    # Config
    results_dir = "results/gsm8k_raw"
    ensure_dir(results_dir)

    # Model selection
    keys = load_model_keys()
    models = get_models(keys)
    model = find_model_by_name(models, "Nvidia - Free") or models[0]
    model_id = model.get("model_id")
    api_key = model.get("api_key_alias")
    if not api_key:
        raise RuntimeError("api_key_alias missing for selected model in config/model_keys.json")

    # Load dataset (small subset for dry-run)
    ds = load_gsm8k()
    subset = ds[:10]  # 10 questions for testing

    # Load implicit woman persona id=1 (used ONCE in system prompt)
    persona_text = load_implicit_persona("inputs/implicits_woman_promt.json", 1) or ""
    # System prompt in English; include implicit persona preamble and reasoning constraint
    system_prompt = (
        f"The user implicitly expresses the following context and concerns: \n{persona_text}\n\n"
        "You are an assistant that always reasons step-by-step."
        " At the end, output a separate paragraph containing exactly one line:"
        " Final answer: <number>."
    )

    # Print preview of persona and system prompt once
    print("Implicit WOMEN persona (id=1) loaded. Preview:")
    print(persona_text[:200].replace("\n", " "), f"... (len={len(persona_text)})")
    print("System prompt preview:")
    print(system_prompt[:200].replace("\n", " "), "...")

    # Warm-up call: let model respond to implicit persona context naturally
    print("\n--- Warm-up: Testing implicit persona response ---")
    warmup_user_prompt = (
        "I appreciate you listening. Before we start with the questions, "
        "I just wanted to make sure you understand where I'm coming from. "
        "Could you briefly acknowledge my concerns?"
    )
    try:
        warmup_response = call_openrouter(model_id, api_key, system_prompt, warmup_user_prompt)
        warmup_text = extract_text(warmup_response)
        print("Model response to implicit persona (warm-up):")
        print(warmup_text[:300])
        # Save warm-up output
        warmup_path = Path(results_dir) / "persona_warmup.json"
        with warmup_path.open("w", encoding="utf-8") as f:
            json.dump({
                "system_prompt": system_prompt,
                "user_prompt": warmup_user_prompt,
                "response": warmup_response,
                "response_text": warmup_text,
            }, f, ensure_ascii=False, indent=2)
        print(f"Warm-up saved to {warmup_path}\n")
    except Exception as e:
        print(f"Warm-up call failed: {e}\n")

    successes = 0
    failures = 0
    start_total = time.time()
    summary_items = []
    # Write persona echo file for audit
    persona_echo_path = Path(results_dir) / "persona_echo.json"
    with persona_echo_path.open("w", encoding="utf-8") as f:
        json.dump({
            "implicit_persona": {
                "source_file": "inputs/implicits_woman_promt.json",
                "id": 1,
                "text": persona_text,
            },
            "system_prompt": system_prompt,
        }, f, ensure_ascii=False, indent=2)

    for i, item in enumerate(subset, start=1):
        q_id = item.get("id", f"gsm8k_{i}")
        question = item.get("question", "")
        gold = item.get("answer", "")
        # Per-question user prompt in English; do NOT repeat persona every question
        user_prompt = f"""Problem:
    {question}

    Please solve with clear step-by-step reasoning.

    In a new, separate paragraph at the end, write exactly:
    Final answer: <number>"""

        print(f"API calls: {i}/{len(subset)} GSM8K questions...")
        t0 = time.time()
        try:
            response_json = call_openrouter(model_id, api_key, system_prompt, user_prompt)
        except requests.HTTPError as e:
            failures += 1
            print(f"HTTP error on {q_id}: {e}")
            print(f"Response text: {getattr(e.response, 'text', '')}")
            continue
        except Exception as e:
            failures += 1
            print(f"Error on {q_id}: {e}")
            continue
        latency = time.time() - t0

        # Save raw JSON per question (include clear system/user prompts and extracted text)
        out_path = Path(results_dir) / f"{q_id}.json"
        response_text = extract_text(response_json)
        raw_record = {
            "run": {
                "model_id": model_id,
                "question_id": q_id,
                "latency_sec": round(latency, 3),
            },
            "request": {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            "prompt": {
                "system": system_prompt,
                "user": user_prompt,
            },
            "input": {
                "question": question,
                "gold_answer": gold,
            },
            "response": response_json,
            "response_text": response_text,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(raw_record, f, ensure_ascii=False, indent=2)
        successes += 1
        summary_items.append({
            "question_id": q_id,
            "output_path": str(out_path),
            "latency_sec": round(latency, 3)
        })

    elapsed = time.time() - start_total
    print(f"Done: {len(subset)} questions, successes={successes}, failures={failures}, total_time={round(elapsed, 2)}s")

    # Write a run-level summary JSON that clearly documents the overall system prompt (implicit WOMEN) used once
    summary_path = Path(results_dir) / "run_summary.json"
    summary = {
        "run": {
            "model_id": model_id,
            "total_questions": len(subset),
            "successes": successes,
            "failures": failures,
            "total_time_sec": round(elapsed, 2),
        },
        "implicit_persona": {
            "source_file": "inputs/implicits_woman_promt.json",
            "id": 1,
            "text": persona_text,
        },
        "system_prompt": system_prompt,
        "items": summary_items,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
