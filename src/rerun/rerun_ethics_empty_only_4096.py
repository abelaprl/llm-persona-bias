"""
Rerun ETHICS JSON files from a plan CSV, but ONLY when current response content is empty.
Uses qwen/qwen3-next-80b-a3b-thinking with max_tokens=4096.

This protects already-fixed JSON files from being overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen3-next-80b-a3b-thinking"
MODEL_ORDER = 32
MAX_TOKENS = 4096


def load_model_key_alias() -> str:
    cfg = BASE_DIR / "config" / "model_keys.json"
    with open(cfg, "r", encoding="utf-8") as f:
        models = json.load(f)
    for m in models:
        if m.get("order") == MODEL_ORDER:
            return str(m.get("api_key_alias", "")).strip()
    raise RuntimeError(f"Model order {MODEL_ORDER} not found in {cfg}")


def call_api(api_key: str, messages: list[dict], max_retries: int = 6) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }

    for retry_count in range(max_retries + 1):
        start = time.time()
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            latency = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                content = ""
                finish_reason = ""
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    content = str(msg.get("content") or "")
                    finish_reason = str(choices[0].get("finish_reason") or "")
                return {
                    "success": True,
                    "api_response": data,
                    "content": content,
                    "finish_reason": finish_reason,
                    "latency": latency,
                    "retry_count": retry_count,
                    "error": "",
                }

            if resp.status_code in {429, 500, 502, 503, 504} and retry_count < max_retries:
                wait_s = 3 * (2 ** retry_count)
                time.sleep(wait_s)
                continue

            return {
                "success": False,
                "api_response": None,
                "content": "",
                "finish_reason": "",
                "latency": latency,
                "retry_count": retry_count,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            }
        except requests.exceptions.Timeout:
            if retry_count < max_retries:
                wait_s = 5 * (2 ** retry_count)
                time.sleep(wait_s)
                continue
            return {
                "success": False,
                "api_response": None,
                "content": "",
                "finish_reason": "",
                "latency": 0.0,
                "retry_count": retry_count,
                "error": "Timeout after retries",
            }
        except Exception as exc:  # noqa: BLE001
            if retry_count < max_retries:
                wait_s = 2 * (2 ** retry_count)
                time.sleep(wait_s)
                continue
            return {
                "success": False,
                "api_response": None,
                "content": "",
                "finish_reason": "",
                "latency": 0.0,
                "retry_count": retry_count,
                "error": str(exc),
            }

    return {
        "success": False,
        "api_response": None,
        "content": "",
        "finish_reason": "",
        "latency": 0.0,
        "retry_count": max_retries,
        "error": "Unexpected retry loop exit",
    }


def get_existing_content(data: dict) -> str:
    response = data.get("response") or {}
    direct = str(response.get("content") or "").strip()
    if direct:
        return direct

    raw = response.get("raw_api_response") or {}
    choices = raw.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun only empty ETHICS responses from plan")
    parser.add_argument(
        "--plan_csv",
        type=str,
        default="result/ETHICS/qwen_qwen3-next-80b-a3b-thinking/rerun_plan_empty_prediction_label_4096.csv",
    )
    parser.add_argument("--delay_sec", type=float, default=1.2)
    parser.add_argument(
        "--log_csv",
        type=str,
        default="result/ETHICS/qwen_qwen3-next-80b-a3b-thinking/rerun_empty_prediction_4096_results.csv",
    )
    args = parser.parse_args()

    plan_path = (BASE_DIR / args.plan_csv).resolve() if not Path(args.plan_csv).is_absolute() else Path(args.plan_csv)
    if not plan_path.exists():
        raise SystemExit(f"Plan not found: {plan_path}")

    api_key = load_model_key_alias()

    with open(plan_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    skip_non_empty = 0
    rerun_success = 0
    rerun_empty_again = 0
    failed = 0
    missing = 0
    no_messages = 0

    log_rows = []

    print("=" * 80)
    print("ETHICS EMPTY-ONLY RERUN")
    print(f"Plan    : {plan_path}")
    print(f"Model   : {MODEL_ID}")
    print(f"Tokens  : {MAX_TOKENS}")
    print(f"Rows    : {total}")
    print("=" * 80)

    for idx, row in enumerate(rows, 1):
        rel_json = (row.get("json_path") or "").strip()
        json_path = (BASE_DIR / rel_json).resolve() if not Path(rel_json).is_absolute() else Path(rel_json)

        if not json_path.exists():
            missing += 1
            print(f"[{idx}/{total}] MISSING {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "missing_json",
                "latency_sec": "",
                "retry_count": "",
                "error": "json_not_found",
            })
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{idx}/{total}] FAIL read_json {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "read_json_failed",
                "latency_sec": "",
                "retry_count": "",
                "error": str(exc)[:200],
            })
            continue

        existing = get_existing_content(data)
        if existing:
            skip_non_empty += 1
            print(f"[{idx}/{total}] SKIP already_non_empty {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "skip_non_empty",
                "latency_sec": "",
                "retry_count": "",
                "error": "",
            })
            continue

        messages = ((data.get("request") or {}).get("messages") or [])
        if not messages:
            no_messages += 1
            print(f"[{idx}/{total}] FAIL no_messages {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "no_messages",
                "latency_sec": "",
                "retry_count": "",
                "error": "request.messages missing",
            })
            continue

        result = call_api(api_key, messages)
        if not result["success"]:
            failed += 1
            print(f"[{idx}/{total}] FAIL api {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "api_failed",
                "latency_sec": f"{result['latency']:.4f}",
                "retry_count": result["retry_count"],
                "error": result["error"],
            })
            time.sleep(max(0.0, args.delay_sec))
            continue

        new_content = str(result.get("content") or "").strip()
        if not new_content:
            rerun_empty_again += 1
            print(f"[{idx}/{total}] EMPTY_AGAIN {rel_json}")
            log_rows.append({
                "json_path": rel_json,
                "status": "empty_again",
                "latency_sec": f"{result['latency']:.4f}",
                "retry_count": result["retry_count"],
                "error": "response_content_empty",
            })
            time.sleep(max(0.0, args.delay_sec))
            continue

        # Replace only when improved (non-empty content).
        data["response"] = {
            "success": True,
            "content": new_content,
            "raw_api_response": result["api_response"],
            "error": None,
            "performance": {
                "latency_sec": result["latency"],
                "tokens": (result["api_response"] or {}).get("usage", {}),
                "retry_count": result["retry_count"],
            },
        }
        md = data.get("metadata") or {}
        md["rerun_timestamp"] = datetime.now().isoformat()
        md["rerun_max_tokens"] = MAX_TOKENS
        md["rerun_mode"] = "empty_only"
        data["metadata"] = md

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        rerun_success += 1
        print(f"[{idx}/{total}] UPDATED {rel_json}")
        log_rows.append({
            "json_path": rel_json,
            "status": "updated_non_empty",
            "latency_sec": f"{result['latency']:.4f}",
            "retry_count": result["retry_count"],
            "error": "",
        })

        time.sleep(max(0.0, args.delay_sec))

    log_path = (BASE_DIR / args.log_csv).resolve() if not Path(args.log_csv).is_absolute() else Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["json_path", "status", "latency_sec", "retry_count", "error"],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print("=" * 80)
    print("DONE")
    print(f"total            : {total}")
    print(f"skip_non_empty   : {skip_non_empty}")
    print(f"updated_non_empty: {rerun_success}")
    print(f"empty_again      : {rerun_empty_again}")
    print(f"missing_json     : {missing}")
    print(f"no_messages      : {no_messages}")
    print(f"failed           : {failed}")
    print(f"log_csv          : {log_path}")


if __name__ == "__main__":
    main()
