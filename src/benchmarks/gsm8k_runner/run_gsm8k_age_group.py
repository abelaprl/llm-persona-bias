from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful AI assistant."
OUTPUT_ROOT = Path("result")

GSM8K_PROMPT_TEMPLATE = "Solve the following math problem:\n\n{question}\n\nAnswer:"

def clean_bpe_artifacts(text: str) -> str:
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = ' '.join(text.split())
    return text


def clean_api_response(api_response: dict) -> dict:
    if not isinstance(api_response, dict):
        return api_response
    api_response = dict(api_response)
    if "choices" in api_response and isinstance(api_response["choices"], list):
        api_response["choices"] = [dict(choice) for choice in api_response["choices"]]
        for choice in api_response["choices"]:
            if isinstance(choice, dict) and "message" in choice:
                choice["message"] = dict(choice["message"])
                if "content" in choice["message"] and isinstance(choice["message"]["content"], str):
                    choice["message"]["content"] = clean_bpe_artifacts(choice["message"]["content"])
                if "reasoning" in choice["message"] and isinstance(choice["message"]["reasoning"], str):
                    choice["message"]["reasoning"] = clean_bpe_artifacts(choice["message"]["reasoning"])
    return api_response

def load_model_config(order: int) -> dict:
    with open("config/model_keys.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    for model in models:
        if model.get("order") == order:
            return model

    raise RuntimeError(f"Model order {order} not found in config/model_keys.json")


def model_slug(model_cfg: dict) -> str:
    return model_cfg["model_id"].replace("/", "_")


def output_base_folder(model_cfg: dict) -> Path:
    return OUTPUT_ROOT / "GMS8K" / "Age_Group_Persona" / model_slug(model_cfg)


def load_gsm8k_samples(limit_samples: int) -> list[dict]:
    with open("data/gsm8k/gsm8k_test.json", "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("data/gsm8k/gsm8k_test.json is empty or invalid")

    if limit_samples > 0:
        rows = rows[:limit_samples]

    for i, row in enumerate(rows, start=1):
        if "_qid" not in row:
            row["_qid"] = i

    return rows


def load_personas() -> dict[str, dict]:
    with open("inputs/age_group_persona.json", "r", encoding="utf-8") as f:
        personas_list = json.load(f)

    personas = {}
    for p in personas_list:
        persona_name = p["persona"]
        ptype = "implicit" if "implicit" in persona_name else "explicit"
        personas[persona_name] = {
            "type": ptype,
            "id": p["id"],
            "seed_prompt": p["prompt"],
            "explicit_prompt": p["prompt"]
        }

    return personas


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
            time.sleep(3 * (2 ** retry_count))
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
            time.sleep(5 * (2 ** retry_count))
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": "Timeout after retries",
            "retry_count": retry_count,
        }
    except Exception as exc:
        if retry_count < max_retries:
            time.sleep(3 * (2 ** retry_count))
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": str(exc),
            "retry_count": retry_count,
        }


def ensure_warmup(model_cfg: dict, persona_name: str, persona_cfg: dict, force: bool = False) -> Path:
    if persona_cfg.get("type") != "implicit":
        raise RuntimeError(f"Warmup is only defined for implicit personas: {persona_name}")

    base = output_base_folder(model_cfg) / persona_name / "warmup"
    base.mkdir(parents=True, exist_ok=True)
    warmup_file = base / "persona_warmup.json"

    if warmup_file.exists() and not force:
        return warmup_file

    seed_prompt = (persona_cfg.get("seed_prompt") or "").strip()
    if not seed_prompt:
        seed_prompt = f"Please respond in a style consistent with persona {persona_name}."

    user_prompt = seed_prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    api_result = call_openrouter_api(model_cfg, messages)
    response_content = ""
    raw_api = None

    if api_result["success"]:
        raw_api = api_result["api_response"]
        raw_api = clean_api_response(raw_api)
        choice = raw_api.get("choices", [{}])[0]
        response_content = (choice.get("message", {}).get("content") or "").strip()
        response_content = clean_bpe_artifacts(response_content)

    warmup_json = {
        "metadata": {
            "run_id": f"{model_slug(model_cfg)}_{persona_name}_warmup",
            "experiment_type": "warmup_persona",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg.get("name", ""),
                "api_endpoint": OPENROUTER_URL,
            },
        },
        "input": {
            "persona_name": persona_name,
            "persona_type": persona_cfg["type"],
            "prompt": user_prompt,
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_cfg["model_id"],
            },
        },
        "response": {
            "success": api_result["success"],
            "content": response_content if api_result["success"] else None,
            "raw_api_response": raw_api,
            "error": api_result["error"],
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": (raw_api or {}).get("usage", {}),
                "retry_count": api_result["retry_count"],
            },
        },
    }

    with open(warmup_file, "w", encoding="utf-8") as f:
        json.dump(warmup_json, f, ensure_ascii=False, indent=2)

    return warmup_file


def load_warmup(model_cfg: dict, persona_name: str) -> dict | None:
    warmup_file = output_base_folder(model_cfg) / persona_name / "warmup" / "persona_warmup.json"
    if not warmup_file.exists():
        return None
    with open(warmup_file, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_warmup_data(warmup_data: dict | None) -> bool:
    if not isinstance(warmup_data, dict):
        return False
    warmup_q = (warmup_data.get("input", {}) or {}).get("prompt", "")
    warmup_a = (warmup_data.get("response", {}) or {}).get("content", "")
    return bool(str(warmup_q).strip()) and bool(str(warmup_a).strip())


def build_messages(persona_name: str, persona_cfg: dict, question: str, warmup_data: dict | None) -> list[dict] | None:
    ptype = persona_cfg["type"]
    prompt = GSM8K_PROMPT_TEMPLATE.format(question=question)

    if ptype == "implicit":
        if not warmup_data:
            return None
        warmup_q = warmup_data.get("input", {}).get("prompt", "")
        warmup_a = warmup_data.get("response", {}).get("content", "")
        if not warmup_q or not warmup_a:
            return None
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": warmup_q},
            {"role": "assistant", "content": warmup_a},
            {"role": "user", "content": prompt},
        ]

    if ptype == "explicit":
        explicit_prompt = (persona_cfg.get("explicit_prompt") or "").strip()
        if not explicit_prompt:
            return None
        combined_user = f"{explicit_prompt}\n\n{prompt}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_user},
        ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def create_result_json(model_cfg: dict, persona_name: str, persona_cfg: dict, sample: dict, messages: list[dict], api_result: dict) -> dict:
    question_id = int(sample.get("idx") or sample.get("_qid") or 0)
    run_id = f"{model_slug(model_cfg)}_{persona_name}_gsm8k_{question_id:05d}"

    persona_payload = {"type": persona_name}
    if persona_cfg["type"] == "implicit":
        warmup_prompt = ""
        warmup_response = ""
        if len(messages) >= 3:
            warmup_prompt = messages[1].get("content", "")
            warmup_response = messages[2].get("content", "")
        persona_payload.update(
            {
                "id": persona_cfg["id"],
                "warmup_prompt": warmup_prompt,
                "warmup_response": warmup_response,
            }
        )
    elif persona_cfg["type"] == "explicit":
        persona_payload.update(
            {
                "id": persona_cfg["id"],
                "explicit_prompt": (persona_cfg.get("explicit_prompt") or "").strip(),
            }
        )

    result = {
        "metadata": {
            "run_id": run_id,
            "experiment_type": "gsm8k_bias_study",
            "scenario": "A",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg.get("name", ""),
                "api_endpoint": OPENROUTER_URL,
            },
        },
        "input": {
            "question_id": question_id,
            "dataset": "gsm8k_test",
            "question": sample.get("question", ""),
            "ground_truth": sample.get("answer", ""),
            "persona": persona_payload,
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_cfg["model_id"],
            },
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
        },
    }

    if api_result["success"]:
        api_resp = api_result["api_response"]
        api_resp = clean_api_response(api_resp)
        choice = api_resp.get("choices", [{}])[0]
        content = (choice.get("message", {}).get("content") or "").strip()
        content = clean_bpe_artifacts(content)
        result["response"] = {
            "success": True,
            "content": content,
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": api_resp.get("usage", {}),
                "retry_count": api_result["retry_count"],
            },
        }
    else:
        result["response"] = {
            "success": False,
            "content": None,
            "raw_api_response": None,
            "error": api_result["error"],
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": {},
                "retry_count": api_result["retry_count"],
            },
        }

    return result


def save_result_json(model_cfg: dict, persona_name: str, question_id: int, result_json: dict) -> Path:
    folder = output_base_folder(model_cfg) / persona_name
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"gsm8k_{question_id:05d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM8K test for age group personas on a single model")
    parser.add_argument("--model_order", type=int, default=43)
    parser.add_argument("--limit_samples", type=int, default=1)
    parser.add_argument("--force_warmup", action="store_true")
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help="Comma-separated persona names to run (default: all age group personas)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip GSM8K files that already exist")
    parser.add_argument("--output_root", type=str, default="result", help="Base output root (default: result)")
    parser.add_argument("--sleep_sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    global OUTPUT_ROOT
    args = parse_args()
    OUTPUT_ROOT = Path(args.output_root)

    model_cfg = load_model_config(args.model_order)
    personas = load_personas()
    samples = load_gsm8k_samples(args.limit_samples)

    print("=" * 80)
    print("GSM8K RUNNER - AGE GROUP PERSONAS")
    print("=" * 80)
    print(f"Model: [{args.model_order}] {model_cfg['name']} ({model_cfg['model_id']})")
    print(f"Samples: {len(samples)}")
    if args.personas.strip():
        selected_personas = [p.strip() for p in args.personas.split(",") if p.strip()]
        invalid = [p for p in selected_personas if p not in personas]
        if invalid:
            raise RuntimeError(f"Invalid persona(s): {', '.join(invalid)}")
    else:
        selected_personas = list(personas.keys())

    print(f"Personas: {len(selected_personas)}")
    print(f"Output base: {output_base_folder(model_cfg)}")

    implicit_personas = [p for p in selected_personas if personas[p]["type"] == "implicit"]

    if implicit_personas:
        print("\n[1/2] Generating warmup for implicit personas only...")
        for persona_name in implicit_personas:
            p_cfg = personas[persona_name]
            warmup_path = ensure_warmup(model_cfg, persona_name, p_cfg, force=args.force_warmup)
            warmup_data = load_warmup(model_cfg, persona_name)
            if not is_valid_warmup_data(warmup_data):
                print(f"  [WARN] {persona_name} warmup invalid, regenerating...")
                warmup_path = ensure_warmup(model_cfg, persona_name, p_cfg, force=True)
                warmup_data = load_warmup(model_cfg, persona_name)
                if not is_valid_warmup_data(warmup_data):
                    raise RuntimeError(f"Warmup generation failed for {persona_name}; stop to avoid cascading FAIL files")
            print(f"  [OK] {persona_name} -> {warmup_path}")
            time.sleep(max(args.sleep_sec, 0.0))

    print("\n[2/2] Running GSM8K per persona...")
    total = len(selected_personas) * len(samples)
    task_i = 0
    for persona_name in selected_personas:
        p_cfg = personas[persona_name]
        warmup = load_warmup(model_cfg, persona_name) if p_cfg["type"] == "implicit" else None
        if p_cfg["type"] == "implicit" and not is_valid_warmup_data(warmup):
            print(f"[WARN] {persona_name} warmup missing/invalid during run, regenerating...")
            ensure_warmup(model_cfg, persona_name, p_cfg, force=True)
            warmup = load_warmup(model_cfg, persona_name)
            if not is_valid_warmup_data(warmup):
                raise RuntimeError(f"Warmup still invalid for {persona_name}; aborting run")
        print(f"\n[PERSONA] {persona_name} ({p_cfg['type']})")

        for sample in samples:
            task_i += 1
            qid = int(sample.get("idx") or sample.get("_qid") or 0)
            print(f"  [{task_i}/{total}] gsm8k_{qid:05d} ...", end=" ", flush=True)

            out_path = output_base_folder(model_cfg) / persona_name / f"gsm8k_{qid:05d}.json"
            if args.skip_existing and out_path.exists():
                print("[SKIP] exists")
                continue

            messages = build_messages(persona_name, p_cfg, sample.get("question", ""), warmup)
            if not messages:
                print("[FAIL] cannot build messages")
                fail_result = create_result_json(
                    model_cfg=model_cfg,
                    persona_name=persona_name,
                    persona_cfg=p_cfg,
                    sample=sample,
                    messages=[],
                    api_result={
                        "success": False,
                        "api_response": None,
                        "latency": 0.0,
                        "error": "Cannot build persona messages",
                        "retry_count": 0,
                    },
                )
                out_path = save_result_json(model_cfg, persona_name, qid, fail_result)
                print(f"-> {out_path}")
                continue

            api_result = call_openrouter_api(model_cfg, messages)
            result_json = create_result_json(model_cfg, persona_name, p_cfg, sample, messages, api_result)
            out_path = save_result_json(model_cfg, persona_name, qid, result_json)

            if api_result["success"]:
                print(f"[OK] -> {out_path}")
            else:
                print(f"[FAIL] -> {out_path}")

            time.sleep(max(args.sleep_sec, 0.0))

    print("\nDone.")


if __name__ == "__main__":
    main()
