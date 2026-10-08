import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import requests


CONFIG_PATH = Path("config/model_keys.json")
DATA_PATH = Path("data/gsm8k/gsm8k_test.json")
RESULT_ROOT = Path("result")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def model_folder_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "-")


def load_model_order(order: int) -> dict:
    models = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for model in models:
        if model.get("order") == order:
            return model
    raise ValueError(f"Model dengan order {order} tidak ditemukan di {CONFIG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1 neutral GSM8K result file")
    parser.add_argument("--model-order", type=int, required=True, help="Model order dari config/model_keys.json")
    parser.add_argument("--question-id", type=int, default=0, help="Question index (default: 0)")
    parser.add_argument("--max-retries", type=int, default=0, help="Max retry jika gagal. 0 = retry terus sampai sukses")
    parser.add_argument("--retry-delay", type=int, default=3, help="Jeda retry dalam detik")
    args = parser.parse_args()

    model = load_model_order(args.model_order)

    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sample = dataset[args.question_id]
    question = sample["question"]
    ground_truth = sample.get("answer", "")

    neutral_prompt = "Solve the following question:"
    user_content = f"{neutral_prompt} {question}\n\nAnswer:"

    model_id = model["model_id"]
    model_name = model.get("name", model_id)
    folder_name = model_folder_name(model_id)

    out_dir = RESULT_ROOT / folder_name / "neutral"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gsm8k_{args.question_id:05d}.json"

    # Skip jika file sudah ada
    if out_path.exists():
        print(f"⊘ Model {args.model_order}, Q{args.question_id:04d} | Already exists, skipping")
        return

    result = {
        "metadata": {
            "run_id": f"{folder_name}_neutral_gsm8k_{args.question_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "scenario": "A",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_id,
                "model_name": model_name,
                "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
            },
        },
        "input": {
            "question_id": args.question_id,
            "dataset": "gsm8k_test",
            "question": question,
            "ground_truth": ground_truth,
            "persona": {
                "id": 0,
                "type": "neutral"            },
        },
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_content},
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_id,
            },
        },
        "response": {
            "success": False,
            "content": "",
            "raw_api_response": None,
            "latency": None,
        },
    }

    headers = {
        "Authorization": f"Bearer {model['api_key_alias']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": result["request"]["messages"],
        "temperature": result["request"]["parameters"]["temperature"],
        "max_tokens": result["request"]["parameters"]["max_tokens"],
    }

    retry_count = 0
    success = False

    while not success:
        start = time.time()
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            latency = time.time() - start
            result["response"]["latency"] = latency

            if resp.status_code == 200:
                api_json = resp.json()
                content = (
                    api_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                result["response"]["success"] = True
                result["response"]["content"] = content
                result["response"]["raw_api_response"] = api_json
                success = True

                # Extract & print token info
                usage = api_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                print(f"✓ Model {args.model_order}, Q{args.question_id:04d} | Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            else:
                retry_count += 1
                result["response"]["success"] = False
                result["response"]["content"] = ""
                result["response"]["raw_api_response"] = {
                    "status_code": resp.status_code,
                    "text": resp.text[:200],
                }
                if args.max_retries > 0 and retry_count >= args.max_retries:
                    break
                print(f"⚠ Model {args.model_order}, Q{args.question_id:04d} | HTTP {resp.status_code}, retry in {args.retry_delay}s... (attempt {retry_count})")
                time.sleep(args.retry_delay)
        except Exception as exc:
            retry_count += 1
            result["response"]["success"] = False
            result["response"]["content"] = ""
            result["response"]["raw_api_response"] = {"error": str(exc)[:200]}
            result["response"]["latency"] = time.time() - start
            if args.max_retries > 0 and retry_count >= args.max_retries:
                break
            print(f"⚠ Model {args.model_order}, Q{args.question_id:04d} | Error: {str(exc)[:50]}, retry in {args.retry_delay}s... (attempt {retry_count})")
            time.sleep(args.retry_delay)

    if not success:
        print(f"✗ Model {args.model_order}, Q{args.question_id:04d} | Failed after {retry_count} retries")
    
    result["retry_count"] = retry_count

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ Saved: {out_path}")

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
