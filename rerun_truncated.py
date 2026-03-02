"""
Rerun truncated/no-extract cases from rerun_plan.csv using existing messages.
Uses model order 31 (openai/gpt-oss-20b) with max_tokens=2048.
"""
import json
import csv
import time
import requests
from pathlib import Path
from datetime import datetime

MAX_TOKENS = 2048
TARGET_MODEL_ID = "openai/gpt-oss-20b"
RERUN_PLAN = "rerun_plan.csv"
BASE_DIR = Path(__file__).parent
DELAY_SECONDS = 1.5


def load_order31_key():
    with open(BASE_DIR / "config" / "model_keys.json", encoding="utf-8") as f:
        models = json.load(f)
    for m in models:
        if m.get("order") == 31:
            return m["api_key_alias"]
    raise ValueError("Order 31 not found in model_keys.json")


def call_api(api_key, messages, max_tokens=MAX_TOKENS, max_retries=3):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": TARGET_MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    retry_count = 0
    while retry_count <= max_retries:
        start = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            latency = time.time() - start
            if resp.status_code == 200:
                return {"success": True, "data": resp.json(), "latency": latency,
                        "retry_count": retry_count}
            elif resp.status_code in [429, 500, 502, 503, 504]:
                wait = 3 * (2 ** retry_count)
                print(f"      [RETRY] HTTP {resp.status_code} — waiting {wait}s...")
                time.sleep(wait)
                retry_count += 1
            else:
                return {"success": False, "data": None, "latency": latency,
                        "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        except requests.exceptions.Timeout:
            wait = 5 * (2 ** retry_count)
            print(f"      [RETRY] Timeout — waiting {wait}s...")
            time.sleep(wait)
            retry_count += 1
        except Exception as e:
            return {"success": False, "data": None, "latency": time.time() - start, "error": str(e)}
    return {"success": False, "data": None, "latency": 0, "error": "Max retries exceeded"}


def main():
    api_key = load_order31_key()
    print(f"Model : {TARGET_MODEL_ID}")
    print(f"Key   : {api_key[:12]}...")
    print(f"Tokens: {MAX_TOKENS}")
    print("=" * 70)

    with open(BASE_DIR / RERUN_PLAN, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    success = failed = 0

    for i, row in enumerate(rows, 1):
        json_rel = row["json_path"].replace("//", "/")
        json_path = BASE_DIR / json_rel

        if not json_path.exists():
            print(f"[{i}/{total}] MISSING: {json_rel}")
            failed += 1
            continue

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("request", {}).get("messages")
        if not messages:
            print(f"[{i}/{total}] NO MESSAGES in {json_rel}")
            failed += 1
            continue

        q_id = data.get("input", {}).get("question_id", "?")
        print(f"[{i}/{total}] q={q_id:>5}  {json_rel}  ...", end=" ", flush=True)

        result = call_api(api_key, messages, MAX_TOKENS)

        if result["success"]:
            api_resp = result["data"]
            choice = api_resp["choices"][0]
            content = choice["message"].get("content", "")
            finish_reason = choice.get("finish_reason", "")

            # Update response section in-place
            data["response"] = {
                "success": True,
                "content": content,
                "raw_api_response": api_resp,
                "error": None,
                "performance": {
                    "latency_sec": result["latency"],
                    "tokens": api_resp.get("usage", {}),
                    "retry_count": result["retry_count"],
                },
            }
            data["metadata"]["rerun_timestamp"] = datetime.now().isoformat()
            data["metadata"]["rerun_max_tokens"] = MAX_TOKENS

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            tokens = api_resp.get("usage", {}).get("total_tokens", "?")
            print(f"OK  finish={finish_reason}  tokens={tokens}  {result['latency']:.2f}s")
            success += 1
        else:
            print(f"FAIL  {result.get('error', '')[:80]}")
            failed += 1

        time.sleep(DELAY_SECONDS)

    print("=" * 70)
    print(f"Done: {success} success, {failed} failed / {total} total")


if __name__ == "__main__":
    main()
