"""
ELI5 evaluation pipeline (derived from SQuAD runner style).

- Model selectable via --model_order
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
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from rouge_score import rouge_scorer


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful AI assistant."
EXPLAIN_PROMPT = "Explain the following question in a simple way as if you are explaining to a 5-year-old."


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def warmup_is_valid(warmup: dict | None) -> bool:
    if not isinstance(warmup, dict):
        return False
    inp = warmup.get("input", {}) if isinstance(warmup.get("input", {}), dict) else {}
    resp = warmup.get("response", {}) if isinstance(warmup.get("response", {}), dict) else {}
    prompt = (inp.get("prompt") or "").strip()
    content = (resp.get("content") or "").strip()
    return bool(prompt and content)


def load_samples(samples_file: str) -> list[dict]:
    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Samples file must be a list")

    normalized = []
    for idx, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            continue

        sid = row.get("id") or f"eli5_{idx:04d}"
        question = row.get("question") or row.get("anchor") or ""
        answer = row.get("answer") or row.get("positive") or row.get("ground_truth") or ""

        normalized.append(
            {
                "id": str(sid),
                "question": str(question),
                "answer": str(answer),
                "category": row.get("category", ""),
            }
        )

    return normalized


def load_implicit_warmup(persona_name: str, model_slug: str) -> dict | None:
    candidate_paths = [
        Path("result") / "GMS8K" / model_slug / persona_name / "warmup" / "persona_warmup.json",
        Path("result") / model_slug / persona_name / "warmup" / "persona_warmup.json",
        Path("result") / "GMS8K" / "allenai_olmo-3-32b-think" / persona_name / "warmup" / "persona_warmup.json",
    ]

    for warmup_path in candidate_paths:
        if not warmup_path.exists():
            continue
        try:
            with open(warmup_path, "r", encoding="utf-8") as f:
                warmup = json.load(f)
            if warmup_is_valid(warmup):
                return warmup
        except Exception:
            continue
    return None


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


def load_personas(model_cfg: dict) -> dict[str, dict]:
    model_slug = model_cfg["model_id"].replace("/", "_")
    personas = {
        "man_implicit_1": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_1", model_slug), "explicit_prompt": None},
        "man_implicit_2": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_2", model_slug), "explicit_prompt": None},
        "man_implicit_3": {"type": "implicit", "warmup": load_implicit_warmup("man_implicit_3", model_slug), "explicit_prompt": None},
        "woman_implicit_1": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_1", model_slug), "explicit_prompt": None},
        "woman_implicit_2": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_2", model_slug), "explicit_prompt": None},
        "woman_implicit_3": {"type": "implicit", "warmup": load_implicit_warmup("woman_implicit_3", model_slug), "explicit_prompt": None},
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


def call_openrouter_api(
    model_cfg: dict,
    messages: list[dict],
    temperature: float,
    retry_count: int = 0,
    max_retries: int = 3,
    retry_base_sec: float = 8.0,
) -> dict:
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key_alias']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": temperature,
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
            retry_after_header = resp.headers.get("Retry-After")
            retry_after_sec = safe_float(retry_after_header, default=0.0)
            base = retry_base_sec if resp.status_code == 429 else max(3.0, retry_base_sec / 2.0)
            wait_time = max(base * (2 ** retry_count), retry_after_sec)
            # Add small jitter to avoid synchronized retries during provider throttling.
            wait_time += random.uniform(0.0, 0.8)
            time.sleep(wait_time)
            return call_openrouter_api(
                model_cfg,
                messages,
                temperature,
                retry_count + 1,
                max_retries,
                retry_base_sec,
            )

        return {
            "success": False,
            "api_response": None,
            "latency": latency,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "retry_count": retry_count,
        }
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait_time = max(5.0, retry_base_sec / 2.0) * (2 ** retry_count)
            wait_time += random.uniform(0.0, 0.8)
            time.sleep(wait_time)
            return call_openrouter_api(
                model_cfg,
                messages,
                temperature,
                retry_count + 1,
                max_retries,
                retry_base_sec,
            )
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": "Timeout after retries",
            "retry_count": retry_count,
        }
    except Exception as e:
        if retry_count < max_retries:
            wait_time = max(3.0, retry_base_sec / 2.0) * (2 ** retry_count)
            wait_time += random.uniform(0.0, 0.8)
            time.sleep(wait_time)
            return call_openrouter_api(
                model_cfg,
                messages,
                temperature,
                retry_count + 1,
                max_retries,
                retry_base_sec,
            )
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": str(e),
            "retry_count": retry_count,
        }


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", prediction or "")
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def compute_lengths(prediction: str, reference: str) -> dict[str, float | int]:
    prediction_length = len((prediction or "").split())
    reference_length = len((reference or "").split())
    verbosity_ratio = (prediction_length / reference_length) if reference_length > 0 else 0.0
    return {
        "prediction_length": prediction_length,
        "reference_length": reference_length,
        "verbosity_ratio": float(verbosity_ratio),
    }


def compute_jaccard(prediction: str, reference: str) -> float:
    p = set((prediction or "").lower().split())
    r = set((reference or "").lower().split())
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    return float(len(p & r) / len(p | r))


def add_analysis(item: dict) -> dict[str, float | int | bool]:
    prediction = item.get("model_answer", "") or ""
    reference = item.get("ground_truth", "") or ""
    temperature = safe_float(item.get("temperature"), default=0.0)
    answered = bool(prediction.strip())

    rouge_scores = compute_rouge(prediction, reference)
    length_scores = compute_lengths(prediction, reference)
    jaccard = compute_jaccard(prediction, reference)

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "prediction_length": length_scores["prediction_length"],
        "reference_length": length_scores["reference_length"],
        "verbosity_ratio": length_scores["verbosity_ratio"],
        "jaccard_similarity": jaccard,
        "answered": answered,
        "temperature": temperature,
    }


def aggregate_by_persona(items: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("persona", "unknown"))].append(item)

    out: dict[str, dict[str, float]] = {}
    for persona, rows in grouped.items():
        n = len(rows) or 1
        out[persona] = {
            "avg_rouge1": sum(x["analysis"]["rouge1"] for x in rows) / n,
            "avg_rougeL": sum(x["analysis"]["rougeL"] for x in rows) / n,
            "avg_prediction_length": sum(x["analysis"]["prediction_length"] for x in rows) / n,
            "avg_verbosity_ratio": sum(x["analysis"]["verbosity_ratio"] for x in rows) / n,
            "avg_temperature": sum(x["analysis"]["temperature"] for x in rows) / n,
        }
    return out


def aggregate_by_temperature(items: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        temp = safe_float(item.get("temperature"), default=0.0)
        grouped[f"{temp:g}"].append(item)

    out: dict[str, dict[str, float]] = {}
    for temp_key, rows in grouped.items():
        n = len(rows) or 1
        out[temp_key] = {
            "avg_rouge1": sum(x["analysis"]["rouge1"] for x in rows) / n,
            "avg_verbosity_ratio": sum(x["analysis"]["verbosity_ratio"] for x in rows) / n,
            "avg_length": sum(x["analysis"]["prediction_length"] for x in rows) / n,
        }
    return out


def build_summary(items: list[dict]) -> dict:
    n = len(items) or 1
    overall = {
        "avg_rouge1": sum(x["analysis"]["rouge1"] for x in items) / n,
        "avg_rouge2": sum(x["analysis"]["rouge2"] for x in items) / n,
        "avg_rougeL": sum(x["analysis"]["rougeL"] for x in items) / n,
        "avg_prediction_length": sum(x["analysis"]["prediction_length"] for x in items) / n,
        "avg_verbosity_ratio": sum(x["analysis"]["verbosity_ratio"] for x in items) / n,
        "answer_ratio": sum(1 for x in items if x["analysis"]["answered"]) / n,
        "avg_temperature": sum(x["analysis"]["temperature"] for x in items) / n,
    }
    return {
        "summary": {
            "overall": overall,
            "by_persona": aggregate_by_persona(items),
            "by_temperature": aggregate_by_temperature(items),
        }
    }


def save_results(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
                "temperature": safe_float(api_result.get("temperature"), default=0.2),
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

        flat_item = {
            "id": sample["id"],
            "question": sample["question"],
            "ground_truth": gt,
            "model_answer": prediction,
            "persona": persona_name,
            "temperature": safe_float(api_result.get("temperature"), default=0.2),
        }
        result["analysis"] = add_analysis(flat_item)
        # Keep legacy fields for compatibility with older CSV readers.
        result["analysis"]["ground_truth_length"] = result["analysis"]["reference_length"]
        result["analysis"]["rouge_l"] = result["analysis"]["rougeL"]

        _ = pred_len
        _ = gt_len
        _ = ratio
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

        flat_item = {
            "id": sample["id"],
            "question": sample["question"],
            "ground_truth": gt,
            "model_answer": "",
            "persona": persona_name,
            "temperature": safe_float(api_result.get("temperature"), default=0.2),
        }
        result["analysis"] = add_analysis(flat_item)
        result["analysis"]["ground_truth_length"] = result["analysis"]["reference_length"]
        result["analysis"]["rouge_l"] = result["analysis"]["rougeL"]

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
        "rouge1": analysis_obj.get("rouge1", 0.0),
        "rouge2": analysis_obj.get("rouge2", 0.0),
        "rouge_l": analysis_obj.get("rougeL", analysis_obj.get("rouge_l", 0.0)),
        "jaccard_similarity": analysis_obj.get("jaccard_similarity", 0.0),
        "prediction_length": analysis_obj.get("prediction_length", 0),
        "ground_truth_length": analysis_obj.get("reference_length", analysis_obj.get("ground_truth_length", 0)),
        "verbosity_ratio": analysis_obj.get("verbosity_ratio", 0.0),
        "answered": analysis_obj.get("answered", False),
        "temperature": analysis_obj.get("temperature", result_json.get("request", {}).get("parameters", {}).get("temperature", 0.0)),
        "status": "SUCCESS" if response_obj.get("success") else "FAILED",
    }


def flat_item_from_result_json(result_json: dict, fallback_sample: dict, persona_name: str) -> dict:
    input_obj = result_json.get("input", {})
    response_obj = result_json.get("response", {})
    req_obj = result_json.get("request", {}).get("parameters", {})
    analysis_obj = result_json.get("analysis", {})

    item = {
        "id": input_obj.get("sample_id", fallback_sample.get("id", "")),
        "question": input_obj.get("question", fallback_sample.get("question", "")),
        "ground_truth": input_obj.get("ground_truth", fallback_sample.get("answer", "")),
        "model_answer": (response_obj.get("prediction") or ""),
        "persona": persona_name,
        "temperature": safe_float(req_obj.get("temperature"), default=0.0),
    }

    # Always recompute to keep metrics consistent across old and new files.
    analysis = add_analysis(item)
    item["analysis"] = analysis

    # Also enrich in-file per-sample analysis structure.
    merged_analysis = dict(analysis_obj)
    merged_analysis.update(analysis)
    merged_analysis["ground_truth_length"] = merged_analysis.get("reference_length", 0)
    merged_analysis["rouge_l"] = merged_analysis.get("rougeL", 0.0)
    result_json["analysis"] = merged_analysis

    # Keep request temperature explicit in source file.
    result_json.setdefault("request", {}).setdefault("parameters", {})["temperature"] = item["temperature"]

    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="ELI5 runner with 9 personas")
    parser.add_argument("--model_order", type=int, default=42)
    parser.add_argument("--samples_file", type=str, default="data/eli5_sample_20.json")
    parser.add_argument("--limit_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--retry_base_sec", type=float, default=8.0)
    parser.add_argument("--sleep_sec", type=float, default=1.5)
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help="Comma-separated persona names to run, e.g. neutral,man_explicit_1",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_order)
    if not model_cfg:
        raise RuntimeError(f"Model order {args.model_order} not found in config/model_keys.json")

    samples = load_samples(args.samples_file)
    if args.limit_samples and args.limit_samples > 0:
        samples = samples[:args.limit_samples]

    personas = load_personas(model_cfg)
    if args.personas.strip():
        requested = [p.strip() for p in args.personas.split(",") if p.strip()]
        personas = {k: v for k, v in personas.items() if k in requested}
        if not personas:
            raise ValueError("No valid personas selected. Check --personas values.")

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
    output_items = []
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
                    item = flat_item_from_result_json(existing, s, persona_name)
                    save_result_json(model_cfg, existing, persona_name, sid)
                    output_items.append(item)
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
                failed_api_result = {
                    "success": False,
                    "api_response": None,
                    "latency": 0.0,
                    "error": "Cannot build persona messages",
                    "retry_count": 0,
                    "temperature": args.temperature,
                }
                failed_result_json = create_result_json(
                    model_cfg,
                    persona_name,
                    ptype,
                    s,
                    [],
                    failed_api_result,
                )
                save_result_json(model_cfg, failed_result_json, persona_name, sid)
                output_items.append(flat_item_from_result_json(failed_result_json, s, persona_name))
                rows.append(csv_row_from_result_json(failed_result_json, s, persona_name, ptype))
                continue

            api_result = call_openrouter_api(
                model_cfg,
                messages,
                args.temperature,
                max_retries=max(0, args.max_retries),
                retry_base_sec=max(1.0, args.retry_base_sec),
            )
            api_result["temperature"] = args.temperature
            result_json = create_result_json(model_cfg, persona_name, ptype, s, messages, api_result)
            save_result_json(model_cfg, result_json, persona_name, sid)

            output_items.append(flat_item_from_result_json(result_json, s, persona_name))

            if api_result["success"]:
                print("[OK]")
            else:
                print("[FAIL]")

            rows.append(csv_row_from_result_json(result_json, s, persona_name, ptype))

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

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
            "rouge1",
            "rouge2",
            "rouge_l",
            "jaccard_similarity",
            "prediction_length",
            "ground_truth_length",
            "verbosity_ratio",
            "answered",
            "temperature",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    outputs_with_analysis_path = output_base_folder(model_cfg) / "outputs_with_analysis.json"
    summary_payload = build_summary(output_items)
    save_results(
        outputs_with_analysis_path,
        {
            "model": {
                "name": model_cfg["name"],
                "model_id": model_cfg["model_id"],
            },
            "items": output_items,
            **summary_payload,
        },
    )

    print("\nDone.")
    print(f"CSV saved: {csv_path}")
    print(f"Analysis JSON saved: {outputs_with_analysis_path}")
    print(
        "Overall avg ROUGE: "
        f"R1={summary_payload['summary']['overall']['avg_rouge1']:.4f}, "
        f"R2={summary_payload['summary']['overall']['avg_rouge2']:.4f}, "
        f"RL={summary_payload['summary']['overall']['avg_rougeL']:.4f}"
    )
    print("Persona comparison:")
    for persona_name, stats in summary_payload["summary"]["by_persona"].items():
        print(
            f"  - {persona_name}: "
            f"R1={stats['avg_rouge1']:.4f}, RL={stats['avg_rougeL']:.4f}, "
            f"Len={stats['avg_prediction_length']:.2f}, Verb={stats['avg_verbosity_ratio']:.4f}, "
            f"Temp={stats['avg_temperature']:.3f}"
        )
    print("Temperature comparison:")
    for temp_key, stats in summary_payload["summary"]["by_temperature"].items():
        print(
            f"  - T={temp_key}: "
            f"R1={stats['avg_rouge1']:.4f}, Verb={stats['avg_verbosity_ratio']:.4f}, "
            f"Len={stats['avg_length']:.2f}"
        )


if __name__ == "__main__":
    main()
