"""
Neutral baseline runner (no persona).
Usage: python src/run_neutral_baseline.py --questions 1-10
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

MODEL_ORDER = 35
PERSONA_ID = 0
PERSONA_TYPE = "neutral"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "gsm8k_test.json"
MODEL_KEYS_PATH = BASE_DIR / "config" / "model_keys.json"
RESULT_BASE = BASE_DIR / "result"


def parse_range(range_str):
    if "-" in range_str:
        start, end = map(int, range_str.split("-"))
        return list(range(start, end + 1))
    if "," in range_str:
        return [int(x.strip()) for x in range_str.split(",")]
    return [int(range_str)]


def load_model_config(order):
    with MODEL_KEYS_PATH.open("r", encoding="utf-8") as f:
        models = json.load(f)
    for model in models:
        if model.get("order") == order:
            return model
    raise ValueError(f"Model order {order} not found in model_keys.json")


def load_gsm8k_question(question_id):
    with DATA_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    for item in dataset:
        if item.get("idx") == question_id or item.get("id") == question_id:
            return item
    if question_id <= len(dataset):
        return dataset[question_id - 1]
    return None


def build_user_message(question):
    return f"Solve the following question: {question}\n\nAnswer:"


def is_api_limit_error(status_code, text):
    if status_code == 429:
        return True
    if status_code == 402:
        return True
    if not text:
        return False
    lower_text = text.lower()
    keywords = ["rate limit", "quota", "too many requests", "insufficient credits", "api limit"]
    return any(k in lower_text for k in keywords)


def call_openrouter_api(model_config, messages):
    headers = {
        "Authorization": f"Bearer {model_config['api_key_alias']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_config["model_id"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    start_time = time.time()
    for attempt in range(2):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time

            if response.status_code == 200:
                return {
                    "success": True,
                    "api_response": response.json(),
                    "latency": latency,
                    "error": None,
                    "retry_count": attempt,
                    "stop_due_to_api_limit": False,
                }

            response_text = response.text[:400] if response.text else ""
            if is_api_limit_error(response.status_code, response_text):
                if attempt == 0:
                    time.sleep(3)
                    continue
                return {
                    "success": False,
                    "api_response": None,
                    "latency": latency,
                    "error": "failed_after_retry_due_to_api_limit",
                    "retry_count": attempt,
                    "stop_due_to_api_limit": True,
                }

            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": f"HTTP {response.status_code}: {response_text}",
                "retry_count": attempt,
                "stop_due_to_api_limit": False,
            }
        except requests.exceptions.Timeout:
            latency = time.time() - start_time
            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": "Timeout",
                "retry_count": attempt,
                "stop_due_to_api_limit": False,
            }
        except Exception as exc:
            latency = time.time() - start_time
            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": f"Exception: {str(exc)}",
                "retry_count": attempt,
                "stop_due_to_api_limit": False,
            }

    return {
        "success": False,
        "api_response": None,
        "latency": time.time() - start_time,
        "error": "Unknown error",
        "retry_count": 1,
        "stop_due_to_api_limit": False,
    }


def create_result_json(model_config, gsm8k_data, messages, api_result):
    model_name = model_config["model_id"].replace("/", "_")
    question_id = gsm8k_data.get("idx", gsm8k_data.get("id", 0))

    content = ""
    if api_result["success"]:
        content = api_result["api_response"]["choices"][0]["message"].get("content", "")

    result = {
        "metadata": {
            "run_id": f"{model_name}_{PERSONA_TYPE}_{PERSONA_ID}_gsm8k_{question_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "scenario": "A",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_config["model_id"],
                "model_name": model_config["name"],
                "api_endpoint": API_URL,
            },
        },
        "input": {
            "question_id": question_id,
            "dataset": "gsm8k_test",
            "question": gsm8k_data["question"],
            "ground_truth": gsm8k_data.get("answer", ""),
            "persona": {
                "id": PERSONA_ID,
                "type": PERSONA_TYPE,
                "explicit_prompt": "",
            },
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_config["model_id"],
            },
        },
        "response": {
            "success": api_result["success"],
            "content": content if api_result["success"] else None,
            "raw_api_response": api_result.get("api_response"),
            "latency": api_result["latency"],
        },
        "retry_count": api_result.get("retry_count", 0),
    }

    if not api_result["success"]:
        result["response"]["error"] = api_result.get("error")

    return result


def save_result(model_config, gsm8k_id, result_json):
    model_name = model_config["model_id"].replace("/", "_")
    folder_path = RESULT_BASE / model_name / "netral"
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / f"gsm8k_{gsm8k_id:05d}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    return file_path


def main():
    parser = argparse.ArgumentParser(description="Neutral baseline GSM8K runner")
    parser.add_argument("--questions", type=str, required=True, help="Question range: 1-10 or 1,2,5")
    args = parser.parse_args()

    question_ids = parse_range(args.questions)

    print("=" * 80)
    print("NEUTRAL BASELINE GSM8K RUNNER")
    print("=" * 80)
    print(f"Model Order: {MODEL_ORDER}")
    print(f"Persona: {PERSONA_TYPE} (id={PERSONA_ID})")
    print(f"Questions: {question_ids[0]}-{question_ids[-1]} ({len(question_ids)} questions)")
    print("=" * 80)

    model_config = load_model_config(MODEL_ORDER)
    print(f"Model: [{model_config['order']}] {model_config['name']}")

    total_tasks = len(question_ids)
    for idx, gsm8k_id in enumerate(question_ids, start=1):
        gsm8k_data = load_gsm8k_question(gsm8k_id)
        if not gsm8k_data:
            print(f"[{idx}/{total_tasks}] Question {gsm8k_id:05d}... [SKIP] Not found")
            continue

        user_message = build_user_message(gsm8k_data["question"])
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_message},
        ]

        api_result = call_openrouter_api(model_config, messages)
        result_json = create_result_json(model_config, gsm8k_data, messages, api_result)
        save_result(model_config, gsm8k_id, result_json)

        if api_result["success"]:
            print(f"[{idx}/{total_tasks}] Question {gsm8k_id:05d}... [OK]")
        else:
            error_text = api_result.get("error", "Unknown")
            print(f"[{idx}/{total_tasks}] Question {gsm8k_id:05d}... [FAIL] {error_text}")

        if api_result.get("stop_due_to_api_limit"):
            print("Stopping run due to API limit after retry.")
            break

        time.sleep(0.5)


if __name__ == "__main__":
    main()
