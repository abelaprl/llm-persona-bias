from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful AI assistant."
OUTPUT_ROOT = Path("result/benchmarks/gsm8k/regional_cultural_persona")
GSM8K_PROMPT_TEMPLATE = "Solve the following math problem:\n\n{question}\n\nAnswer:"


def clean_bpe_artifacts(text: str) -> str:
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = " ".join(text.split())
    return text


_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â\x80")


def fix_encoding(text: str) -> str:
    """Repair mojibake from UTF-8 bytes mis-decoded as Windows-1252 or Latin-1.

    Reverses the corruption by re-encoding in the offending codec and decoding as
    UTF-8. Tries cp1252 first (the common Windows case that produces 'â€"'), then
    latin-1. A repair is only accepted if it removes the mojibake markers, so clean
    text is left untouched.
    """
    if not text or not any(mk in text for mk in _MOJIBAKE_MARKERS):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not any(mk in repaired for mk in _MOJIBAKE_MARKERS):
            return repaired
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
                if "reasoning" in choice["message"] and isinstance(choice["message"].get("reasoning"), str):
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
    return OUTPUT_ROOT / model_slug(model_cfg)


def api_endpoint(model_cfg: dict) -> str:
    return model_cfg.get("endpoint_url") or OPENROUTER_URL


def backend_provider(model_cfg: dict) -> str:
    if model_cfg.get("provider"):
        return str(model_cfg["provider"])
    endpoint = api_endpoint(model_cfg).lower()
    if "localhost" in endpoint or "127.0.0.1" in endpoint:
        return "vllm"
    return "openrouter"


def build_api_headers(model_cfg: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    auth_required = model_cfg.get("auth_required", True)
    api_key = str(model_cfg.get("api_key_alias") or "").strip()
    if auth_required and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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


def load_regional_personas(path: str) -> dict[str, dict]:
    persona_path = Path(path)
    if not persona_path.exists():
        raise RuntimeError(f"Persona file not found: {path}")

    with open(persona_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise RuntimeError(f"Persona file must be a list of objects: {path}")

    personas: dict[str, dict] = {}
    for idx, row in enumerate(rows):
        name = str(row.get("persona") or "").strip()
        prompt = fix_encoding(str(row.get("prompt") or "").strip())
        try:
            pid = int(row.get("id") or 0)
        except (TypeError, ValueError):
            pid = 0

        if not name or not prompt or pid <= 0:
            print(f"  [WARN] dropping persona at index {idx}: missing/invalid id/persona/prompt (id={row.get('id')!r}, persona={row.get('persona')!r})")
            continue

        if "implicit" in name:
            ptype = "implicit"
        elif "explicit" in name:
            ptype = "explicit"
        else:
            ptype = "explicit"
            print(f"  [WARN] persona '{name}' has neither 'implicit' nor 'explicit' in its name; defaulting to explicit (no warmup)")

        if name in personas:
            print(f"  [WARN] duplicate persona name '{name}' at index {idx}; later definition overwrites earlier one")

        personas[name] = {
            "id": pid,
            "type": ptype,
            "seed_prompt": prompt if ptype == "implicit" else None,
            "explicit_prompt": prompt if ptype == "explicit" else None,
        }

    if not personas:
        raise RuntimeError(f"No valid personas found in {path}")

    return personas


def call_openrouter_api(
    model_cfg: dict,
    messages: list[dict],
    retry_count: int = 0,
    max_retries: int = 3,
    request_timeout_sec: float = 60.0,
) -> dict:
    endpoint = api_endpoint(model_cfg)
    headers = build_api_headers(model_cfg)
    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    start = time.time()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=request_timeout_sec)
        latency = time.time() - start

        if resp.status_code == 200:
            try:
                api_response = resp.json()
            except ValueError as exc:
                return {
                    "success": False,
                    "api_response": None,
                    "latency": latency,
                    "error": f"Invalid JSON response: {exc}",
                    "retry_count": retry_count,
                }
            return {
                "success": True,
                "api_response": api_response,
                "latency": latency,
                "error": None,
                "retry_count": retry_count,
            }

        if resp.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            time.sleep(3 * (2 ** retry_count))
            return call_openrouter_api(
                model_cfg, messages, retry_count + 1, max_retries, request_timeout_sec
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
            time.sleep(5 * (2 ** retry_count))
            return call_openrouter_api(
                model_cfg, messages, retry_count + 1, max_retries, request_timeout_sec
            )
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
            return call_openrouter_api(
                model_cfg, messages, retry_count + 1, max_retries, request_timeout_sec
            )
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": str(exc),
            "retry_count": retry_count,
        }


def ensure_warmup(
    model_cfg: dict,
    persona_name: str,
    persona_cfg: dict,
    force: bool = False,
    max_retries: int = 3,
    request_timeout_sec: float = 60.0,
) -> Path:
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": seed_prompt},
    ]

    api_result = call_openrouter_api(
        model_cfg,
        messages,
        max_retries=max_retries,
        request_timeout_sec=request_timeout_sec,
    )
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
                "provider": backend_provider(model_cfg),
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg.get("name", ""),
                "api_endpoint": api_endpoint(model_cfg),
            },
        },
        "input": {
            "persona_name": persona_name,
            "persona_type": persona_cfg["type"],
            "prompt": seed_prompt,
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 4096,
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
    try:
        with open(warmup_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_valid_warmup_data(warmup_data: dict | None) -> bool:
    if not isinstance(warmup_data, dict):
        return False
    response = warmup_data.get("response")
    if not isinstance(response, dict) or response.get("success") is not True:
        return False
    warmup_q = (warmup_data.get("input", {}) or {}).get("prompt", "")
    warmup_a = response.get("content", "")
    return bool(str(warmup_q).strip()) and bool(str(warmup_a).strip())


def build_messages(persona_name: str, persona_cfg: dict, question: str, warmup_data: dict | None) -> list[dict] | None:
    prompt = GSM8K_PROMPT_TEMPLATE.format(question=question)
    ptype = persona_cfg["type"]

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

    explicit_prompt = (persona_cfg.get("explicit_prompt") or "").strip()
    if not explicit_prompt:
        return None
    combined_user = f"{explicit_prompt}\n\n{prompt}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_user},
    ]


def create_result_json(model_cfg: dict, persona_name: str, persona_cfg: dict, sample: dict, messages: list[dict], api_result: dict) -> dict:
    question_id = int(sample.get("idx") or sample.get("_qid") or 0)
    run_id = f"{model_slug(model_cfg)}_{persona_name}_gsm8k_{question_id:05d}"

    persona_payload = {
        "type": persona_name,
        "id": persona_cfg.get("id"),
    }
    if persona_cfg["type"] == "implicit":
        warmup_prompt = ""
        warmup_response = ""
        if len(messages) >= 3:
            warmup_prompt = messages[1].get("content", "")
            warmup_response = messages[2].get("content", "")
        persona_payload.update(
            {
                "warmup_prompt": warmup_prompt,
                "warmup_response": warmup_response,
            }
        )
    else:
        persona_payload.update(
            {
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
                "provider": backend_provider(model_cfg),
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg.get("name", ""),
                "api_endpoint": api_endpoint(model_cfg),
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
                "temperature": 0.0,
                "max_tokens": 4096,
                "model": model_cfg["model_id"],
            },
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
        },
    }

    if api_result["success"]:
        api_resp = clean_api_response(api_result["api_response"])
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


def result_file_has_successful_response(path: Path) -> bool:
    """Treat only readable, technically successful responses as complete."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError):
        return False

    if not isinstance(payload, dict):
        return False
    response = payload.get("response")
    return isinstance(response, dict) and response.get("success") is True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM8K test for regional cultural personas on a single model")
    parser.add_argument("--model_order", type=int, default=43)
    parser.add_argument("--limit_samples", type=int, default=1)
    parser.add_argument("--force_warmup", action="store_true")
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help="Comma-separated persona names to run (default: all personas from file)",
    )
    parser.add_argument("--persona_file", type=str, default="inputs/regional_cultural_persona.json")
    parser.add_argument("--skip_existing", action="store_true", help="Skip GSM8K files that already exist")
    parser.add_argument("--output_root", type=str, default="result/benchmarks/gsm8k/regional_cultural_persona")
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="",
        help="Override the model endpoint, e.g. a local SSH tunnel to vLLM.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request_timeout_sec", type=float, default=60.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    global OUTPUT_ROOT
    args = parse_args()
    OUTPUT_ROOT = Path(args.output_root)

    if args.concurrency < 1:
        raise RuntimeError("--concurrency must be at least 1")
    if args.request_timeout_sec <= 0:
        raise RuntimeError("--request_timeout_sec must be greater than 0")
    if args.max_retries < 0:
        raise RuntimeError("--max_retries cannot be negative")

    model_cfg = load_model_config(args.model_order)
    if args.endpoint_url.strip():
        model_cfg = {
            **model_cfg,
            "endpoint_url": args.endpoint_url.strip(),
            "auth_required": False,
            "provider": "vllm",
        }
    personas = load_regional_personas(args.persona_file)
    samples = load_gsm8k_samples(args.limit_samples)

    if args.personas.strip():
        selected = [p.strip() for p in args.personas.split(",") if p.strip()]
        invalid = [p for p in selected if p not in personas]
        if invalid:
            raise RuntimeError(f"Invalid persona(s): {', '.join(invalid)}")
    else:
        selected = list(personas.keys())

    implicit_personas = [p for p in selected if personas[p]["type"] == "implicit"]

    print("=" * 80)
    print("GSM8K RUNNER - REGIONAL CULTURAL PERSONAS")
    print("=" * 80)
    print(f"Model: [{args.model_order}] {model_cfg['name']} ({model_cfg['model_id']})")
    print(f"Samples: {len(samples)}")
    print(f"Personas: {len(selected)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output base: {output_base_folder(model_cfg)}")

    print("\n[1/2] Generating warmup for implicit personas only...")
    for persona_name in implicit_personas:
        p_cfg = personas[persona_name]
        warmup_path = ensure_warmup(
            model_cfg,
            persona_name,
            p_cfg,
            force=args.force_warmup,
            max_retries=args.max_retries,
            request_timeout_sec=args.request_timeout_sec,
        )
        warmup_data = load_warmup(model_cfg, persona_name)
        if not is_valid_warmup_data(warmup_data):
            print(f"  [WARN] {persona_name} warmup invalid, regenerating...")
            warmup_path = ensure_warmup(
                model_cfg,
                persona_name,
                p_cfg,
                force=True,
                max_retries=args.max_retries,
                request_timeout_sec=args.request_timeout_sec,
            )
            warmup_data = load_warmup(model_cfg, persona_name)
            if not is_valid_warmup_data(warmup_data):
                raise RuntimeError(f"Warmup generation failed for {persona_name}; stop to avoid cascading FAIL files")
        print(f"  [OK] {persona_name} -> {warmup_path}")
        time.sleep(max(args.sleep_sec, 0.0))

    print("\n[2/2] Running GSM8K per persona...")
    total = len(selected) * len(samples)
    task_i = 0
    for persona_name in selected:
        p_cfg = personas[persona_name]
        warmup = load_warmup(model_cfg, persona_name) if p_cfg["type"] == "implicit" else None
        if p_cfg["type"] == "implicit" and not is_valid_warmup_data(warmup):
            print(f"[WARN] {persona_name} warmup missing/invalid during run, regenerating...")
            ensure_warmup(
                model_cfg,
                persona_name,
                p_cfg,
                force=True,
                max_retries=args.max_retries,
                request_timeout_sec=args.request_timeout_sec,
            )
            warmup = load_warmup(model_cfg, persona_name)
            if not is_valid_warmup_data(warmup):
                raise RuntimeError(f"Warmup still invalid for {persona_name}; aborting run")
        print(f"\n[PERSONA] {persona_name} ({p_cfg['type']})")

        pending: list[tuple[int, dict]] = []
        retry_existing = 0
        for sample in samples:
            task_i += 1
            qid = int(sample.get("idx") or sample.get("_qid") or 0)

            out_path = output_base_folder(model_cfg) / persona_name / f"gsm8k_{qid:05d}.json"
            if args.skip_existing and out_path.exists():
                if result_file_has_successful_response(out_path):
                    continue
                retry_existing += 1

            pending.append((task_i, sample))

        skipped = len(samples) - len(pending)
        if skipped:
            print(f"  Existing: {skipped} [SKIP]")
        if retry_existing:
            print(f"  Failed/invalid existing: {retry_existing} [RERUN]")
        if not pending:
            continue

        def run_sample(task: tuple[int, dict]) -> tuple[int, int, Path, bool]:
            task_number, sample = task
            qid = int(sample.get("idx") or sample.get("_qid") or 0)

            messages = build_messages(persona_name, p_cfg, sample.get("question", ""), warmup)
            if not messages:
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
                return task_number, qid, out_path, False

            api_result = call_openrouter_api(
                model_cfg,
                messages,
                max_retries=args.max_retries,
                request_timeout_sec=args.request_timeout_sec,
            )
            result_json = create_result_json(model_cfg, persona_name, p_cfg, sample, messages, api_result)
            out_path = save_result_json(model_cfg, persona_name, qid, result_json)

            time.sleep(max(args.sleep_sec, 0.0))

            return task_number, qid, out_path, api_result["success"]

        def record_result(result: tuple[int, int, Path, bool]) -> None:
            task_number, qid, out_path, success = result
            status = "[OK]" if success else "[FAIL]"
            print(f"  [{task_number}/{total}] gsm8k_{qid:05d} {status} -> {out_path}")

        if args.concurrency == 1:
            for task in pending:
                record_result(run_sample(task))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(run_sample, task) for task in pending]
                for future in as_completed(futures):
                    record_result(future.result())

    print("\nDone.")


if __name__ == "__main__":
    main()
