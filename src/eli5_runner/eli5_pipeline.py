"""
ELI5 evaluation pipeline (derived from SQuAD runner style).

- Model fixed to order 42 (Allen AI: OLMo 3 32B Think)
- Persona support: 9 personas (6 implicit, 2 explicit, 1 neutral)
- Input samples from data/eli5_sample_20.json
- Output JSON: result/ELI5/allenai_olmo-3-32b-think/<persona>/eli5_<id>.json

Usage:
  python src/eli5_runner/eli5_pipeline.py --samples_file data/eli5_sample_20.json --limit_samples 1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful AI assistant."
EXPLAIN_PROMPT = "Explain the answer to the question in a clear and simple paragraph."


def sanitize_user_content(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("Solve the following question:", "")
    cleaned = cleaned.replace("solve the following question:", "")
    return cleaned.strip()


def load_model_config(order: int = 42) -> dict | None:
    with open("config/model_keys.json", "r", encoding="utf-8") as f:
        models = json.load(f)
    for m in models:
        if m.get("order") == order:
            return m
    return None


def load_samples(samples_file: str) -> list[dict]:
    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Samples file must be a list")
    return data


def load_implicit_warmup(persona_name: str) -> dict | None:
    warmup_path = Path(f"result/allenai_olmo-3-32b-think/{persona_name}/warmup/persona_warmup.json")
    if not warmup_path.exists():
        return None
    with open(warmup_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_explicit_persona_prompt(persona_name: str) -> str | None:
    mapping = {
        "man_explicit_1": "inputs/explicits_man_prompts.json",
        "woman_explicit_1": "inputs/explicits_woman_prompts.json",
    }
    file_path = mapping.get(persona_name)
    if not file_path:
        return None

    path = Path(file_path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list) and data:
        return data[0].get("prompt", "")
    if isinstance(data, dict):
        return data.get("prompt", "")
    return None


def load_personas() -> dict[str, dict]:
    personas = {
        "man_implicit_1": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_1"), "explicit_prompt": None},
        "man_implicit_2": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_2"), "explicit_prompt": None},
        "man_implicit_3": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_3"), "explicit_prompt": None},
        "woman_implicit_1": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_1"), "explicit_prompt": None},
        "woman_implicit_2": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_2"), "explicit_prompt": None},
        "woman_implicit_3": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_3"), "explicit_prompt": None},
        "man_explicit_1": {"type": "explicit", "warmup": None, "explicit_prompt": load_explicit_persona_prompt("man_explicit_1")},
        "woman_explicit_1": {"type": "explicit", "warmup": None, "explicit_prompt": load_explicit_persona_prompt("woman_explicit_1")},
        "neutral": {"type": "neutral", "warmup": None, "explicit_prompt": None},
    }
    return personas


def build_messages_implicit(persona_cfg: dict, question: str) -> list[dict] | None:
    warmup = persona_cfg.get("warmup")
    if not warmup:
        return None

    warmup_q = sanitize_user_content(warmup.get("input", {}).get("prompt", ""))
    warmup_a = warmup.get("response", {}).get("content", "")

    user_q = sanitize_user_content(
        f"{EXPLAIN_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": warmup_q},
        {"role": "assistant", "content": warmup_a},
        {"role": "user", "content": user_q},
    ]


def build_messages_explicit(persona_cfg: dict, question: str) -> list[dict] | None:
    persona_prompt = sanitize_user_content(persona_cfg.get("explicit_prompt", ""))
    if not persona_prompt:
        return None

    combined_user = sanitize_user_content(
        f"{persona_prompt}\n\n{EXPLAIN_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_user},
    ]


def build_messages_neutral(question: str) -> list[dict]:
    user_q = sanitize_user_content(
        f"{EXPLAIN_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_q},
    ]


def call_openrouter_api(model_cfg: dict, messages: list[dict], retry_count: int = 0, max_retries: int = 3) -> dict:
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key_alias']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    start = time.time()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start

        if resp.status_code == 200:
            return {
                "success": True,
                "api_response": resp.json(),
                "latency": latency,
                "error": None,
                "retry_count": retry_count,
            }

        if resp.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)

        return {
            "success": False,
            "api_response": None,
            "latency": latency,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "retry_count": retry_count,
        }
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait_time = 5 * (2 ** retry_count)
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": "Timeout after retries",
            "retry_count": retry_count,
        }
    except Exception as e:
        if retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": str(e),
            "retry_count": retry_count,
        }


def lcs_length(a_tokens: list[str], b_tokens: list[str]) -> int:
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def simple_semantic_similarity(prediction: str, ground_truth: str) -> float:
    p = set(re.findall(r"\b\w+\b", prediction.lower()))
    g = set(re.findall(r"\b\w+\b", ground_truth.lower()))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def rouge_l(prediction: str, ground_truth: str) -> float:
    p_tokens = re.findall(r"\b\w+\b", prediction.lower())
    g_tokens = re.findall(r"\b\w+\b", ground_truth.lower())
    if not p_tokens or not g_tokens:
        return 0.0
    lcs = lcs_length(p_tokens, g_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(g_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def create_result_json(model_cfg: dict, persona_name: str, persona_type: str, sample: dict, messages: list[dict], api_result: dict) -> dict:
    timestamp = datetime.now().isoformat()

    result = {
        "metadata": {
            "run_id": f"eli5_{sample['id']}_{persona_name}",
            "experiment_type": "eli5_bias_study",
            "dataset": "ELI5_pair_train_5k",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg["name"],
            },
        },
        "input": {
            "sample_id": sample["id"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "persona": {
                "name": persona_name,
                "type": persona_type,
            },
            "category": sample.get("category", ""),
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_cfg["model_id"],
            },
        },
    }

    if api_result["success"]:
        api_resp = api_result["api_response"]
        choice = api_resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        prediction = (message.get("content") or "").strip()
        gt = sample["answer"]

        pred_len = len(re.findall(r"\b\w+\b", prediction))
        gt_len = len(re.findall(r"\b\w+\b", gt))
        ratio = (pred_len / gt_len) if gt_len > 0 else 0.0

        result["response"] = {
            "success": True,
            "prediction": prediction,
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": api_resp.get("usage", {}),
                "retry_count": api_result["retry_count"],
            },
        }

        result["analysis"] = {
            "semantic_similarity": round(simple_semantic_similarity(prediction, gt), 4),
            "rouge_l": round(rouge_l(prediction, gt), 4),
            "prediction_length": pred_len,
            "ground_truth_length": gt_len,
            "verbosity_ratio": round(ratio, 4),
        }
    else:
        gt = sample["answer"]
        gt_len = len(re.findall(r"\b\w+\b", gt))

        result["response"] = {
            "success": False,
            "prediction": None,
            "raw_api_response": None,
            "error": api_result["error"],
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": {},
                "retry_count": api_result["retry_count"],
            },
        }

        result["analysis"] = {
            "semantic_similarity": 0.0,
            "rouge_l": 0.0,
            "prediction_length": 0,
            "ground_truth_length": gt_len,
            "verbosity_ratio": 0.0,
        }

    return result


def output_base_folder(model_cfg: dict) -> Path:
    model_slug = model_cfg["model_id"].replace("/", "_")
    return Path("result") / "ELI5" / model_slug


def save_result_json(model_cfg: dict, result_json: dict, persona_name: str, sample_id: str) -> str:
    folder = output_base_folder(model_cfg) / persona_name
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{sample_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    return str(filepath)


def load_result_json(model_cfg: dict, persona_name: str, sample_id: str) -> dict | None:
    filepath = output_base_folder(model_cfg) / persona_name / f"{sample_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def result_exists(model_cfg: dict, persona_name: str, sample_id: str) -> bool:
    filepath = output_base_folder(model_cfg) / persona_name / f"{sample_id}.json"
    return filepath.exists()


def csv_row_from_result_json(result_json: dict, fallback_sample: dict, persona_name: str, persona_type: str) -> dict:
    input_obj = result_json.get("input", {})
    response_obj = result_json.get("response", {})
    analysis_obj = result_json.get("analysis", {})
    return {
        "id": input_obj.get("sample_id", fallback_sample.get("id", "")),
        "persona_name": persona_name,
        "persona_type": persona_type,
        "question": input_obj.get("question", fallback_sample.get("question", "")),
        "prediction": response_obj.get("prediction") or "",
        "ground_truth": input_obj.get("ground_truth", fallback_sample.get("answer", "")),
        "category": input_obj.get("category", fallback_sample.get("category", "")),
        "semantic_similarity": analysis_obj.get("semantic_similarity", 0.0),
        "rouge_l": analysis_obj.get("rouge_l", 0.0),
        "prediction_length": analysis_obj.get("prediction_length", 0),
        "ground_truth_length": analysis_obj.get("ground_truth_length", 0),
        "verbosity_ratio": analysis_obj.get("verbosity_ratio", 0.0),
        "status": "SUCCESS" if response_obj.get("success") else "FAILED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ELI5 runner with 9 personas")
    parser.add_argument("--samples_file", type=str, default="data/eli5_sample_20.json")
    parser.add_argument("--limit_samples", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_cfg = load_model_config(42)
    if not model_cfg:
        raise RuntimeError("Model order 42 not found in config/model_keys.json")

    samples = load_samples(args.samples_file)
    if args.limit_samples and args.limit_samples > 0:
        samples = samples[:args.limit_samples]

    personas = load_personas()

    print("=" * 80)
    print("ELI5 EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Samples: {len(samples)}")
    print(f"Personas: {len(personas)}")
    print(f"Total tasks: {len(samples) * len(personas)}")
    print(f"Model: {model_cfg['name']} ({model_cfg['model_id']})")

    if args.dry_run:
        print("Dry run enabled. No API call.")
        return

    rows = []
    task_i = 0

    for persona_name in sorted(personas.keys()):
        persona_cfg = personas[persona_name]
        ptype = persona_cfg["type"]

        print(f"\n[PERSONA] {persona_name} ({ptype})")

        for s in samples:
            task_i += 1
            sid = s["id"]
            print(f"  [{task_i}/{len(samples) * len(personas)}] {sid} ...", end=" ", flush=True)

            if result_exists(model_cfg, persona_name, sid):
                print("[SKIP] exists")
                existing = load_result_json(model_cfg, persona_name, sid)
                if existing:
                    rows.append(csv_row_from_result_json(existing, s, persona_name, ptype))
                continue

            if ptype == "implicit":
                messages = build_messages_implicit(persona_cfg, s["question"])
            elif ptype == "explicit":
                messages = build_messages_explicit(persona_cfg, s["question"])
            else:
                messages = build_messages_neutral(s["question"])

            if not messages:
                print("[FAIL] cannot build messages")
                continue

            api_result = call_openrouter_api(model_cfg, messages)
            result_json = create_result_json(model_cfg, persona_name, ptype, s, messages, api_result)
            save_result_json(model_cfg, result_json, persona_name, sid)

            if api_result["success"]:
                print("[OK]")
            else:
                print("[FAIL]")

            rows.append(csv_row_from_result_json(result_json, s, persona_name, ptype))

            time.sleep(0.5)

    model_slug = model_cfg['model_id'].replace('/', '_')
    csv_path = output_base_folder(model_cfg) / f"result_eli5_{model_slug}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "persona_name",
            "persona_type",
            "question",
            "prediction",
            "ground_truth",
            "category",
            "semantic_similarity",
            "rouge_l",
            "prediction_length",
            "ground_truth_length",
            "verbosity_ratio",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
