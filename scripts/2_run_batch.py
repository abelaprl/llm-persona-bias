import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs_json")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_histories")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_TIMEOUT = 60

def call_openrouter(model, api_key, messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 512
    }

    try:
        start = time.time()
        response = requests.post(API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        latency = time.time() - start
        if response.status_code == 200:
            result = response.json()
            return result, latency
        else:
            return {"error": response.status_code, "message": response.text}, None
    except Exception as e:
        return {"error": "exception", "message": str(e)}, None

# Load model configs
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_configs = json.load(f)

# Run all sessions
for model in model_configs:
    order = model["order"]
    api_key = model["api_key_alias"]
    model_id = model["model_id"]

    input_dir = os.path.join(INPUTS_DIR, f"{order:02d}")
    log_dir = os.path.join(LOGS_DIR, f"{order:02d}")
    hist_dir = os.path.join(HISTORY_DIR, f"{order:02d}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue

        session_path = os.path.join(input_dir, fname)
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)

        run_id = session["run_id"]
        messages = session["messages"]

        result, latency = call_openrouter(model_id, api_key, messages)
        output = {
            "run_id": run_id,
            "model_id": model_id,
            "order": order,
            "question_id": session["question_id"],
            "messages": messages,
            "response": result,
            "latency_sec": latency
        }

        # Save raw output
        with open(os.path.join(log_dir, f"{run_id}.json"), "w", encoding="utf-8") as f_out:
            json.dump(output, f_out, indent=2, ensure_ascii=False)

        # Save chat history only
        with open(os.path.join(hist_dir, f"{run_id}.json"), "w", encoding="utf-8") as f_hist:
            json.dump(messages, f_hist, indent=2, ensure_ascii=False)

        print(f"✅ {run_id} done (latency: {latency:.2f}s)" if latency else f"❌ {run_id} failed")
        time.sleep(0.8)  # basic rate limit