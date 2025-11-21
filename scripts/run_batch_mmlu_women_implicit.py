import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")

# Nama skenario / task
TASK_NAME = "mmlu_women_implicit_strong"

INPUTS_ROOT = os.path.join(BASE_DIR, "inputs", TASK_NAME)
LOGS_ROOT = os.path.join(BASE_DIR, "logs_json", TASK_NAME)
HISTORY_ROOT = os.path.join(BASE_DIR, "chat_histories", TASK_NAME)

os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(HISTORY_ROOT, exist_ok=True)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_TIMEOUT = 60

# Optional filters: set one of these environment variables to run a single model
# - TARGET_MODEL_ID (e.g. meta-llama/llama-3.3-70b-instruct:free)
# - TARGET_ORDER (e.g. 14)
TARGET_MODEL_ID = os.environ.get("TARGET_MODEL_ID")
TARGET_ORDER = os.environ.get("TARGET_ORDER")
# Configurable runtime params
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "3.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.0"))
ENABLE_REASONING = os.environ.get("ENABLE_REASONING", "0") == "1"


def call_openrouter(model, api_key, messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1500,
    }
    # Optionally enable reasoning / chain-of-thought if requested
    if ENABLE_REASONING:
        payload["reasoning"] = "chain_of_thought"

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

print('DEBUG: TARGET_MODEL_ID=', TARGET_MODEL_ID, 'TARGET_ORDER=', TARGET_ORDER)
print('DEBUG: loaded model_configs count=', len(model_configs))

for model in model_configs:
    order = model["order"]
    api_key = model["api_key_alias"]
    model_id = model["model_id"]

    # Apply optional filters
    if TARGET_MODEL_ID and model_id != TARGET_MODEL_ID:
        continue
    # If TARGET_ORDER is set but TARGET_MODEL_ID is also set, we allow
    # running the specific model against a different order's inputs. Only
    # filter by order when TARGET_MODEL_ID is not provided.
    if TARGET_ORDER and not TARGET_MODEL_ID:
        try:
            if int(TARGET_ORDER) != int(order):
                continue
        except Exception:
            pass

    # OPSIONAL: skip model non-OpenRouter (misal groq)
    if model_id.startswith("groq/"):
        print(f"⏭ Skipping model {model_id} (served by Groq, not OpenRouter)")
        continue
    # Allow TARGET_ORDER to override which input folder to use (useful when
    # running a model against inputs from a different order)
    use_order = None
    if TARGET_ORDER:
        try:
            use_order = int(TARGET_ORDER)
        except Exception:
            use_order = None
    if use_order is None:
        use_order = int(order)

    input_dir = os.path.join(INPUTS_ROOT, f"{use_order:02d}")
    if not os.path.isdir(input_dir):
        print(f"⚠️  No input dir for model {order:02d} ({model_id}) at {input_dir}, skipping.")
        continue

    log_dir = os.path.join(LOGS_ROOT, f"{order:02d}")
    hist_dir = os.path.join(HISTORY_ROOT, f"{order:02d}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    print(f"🚀 Running model {order:02d} - {model_id} on inputs from {input_dir}")

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue

        session_path = os.path.join(input_dir, fname)
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)

        run_id = session["run_id"]
        messages = session["messages"]

        # Resume: skip if output already exists
        out_path = os.path.join(log_dir, f"{run_id}.json")
        if os.path.exists(out_path):
            print(f"↪️  Skipping existing output for {run_id}")
            # still save chat history if missing
            hist_path = os.path.join(hist_dir, f"{run_id}.json")
            if not os.path.exists(hist_path):
                with open(hist_path, "w", encoding="utf-8") as f_hist:
                    json.dump(messages, f_hist, indent=2, ensure_ascii=False)
            continue

        # Call with retry/backoff on rate limits
        attempt = 0
        result = None
        latency = None
        while True:
            result, latency = call_openrouter(model_id, api_key, messages)
            # result can be dict with error or API response dict
            if isinstance(result, dict) and result.get("error") == 429:
                attempt += 1
                if attempt <= MAX_RETRIES:
                    backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"429 rate limit for {run_id}; retrying after {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"❌ {run_id} failed after {MAX_RETRIES} retries: {result}")
                    break
            # network/exception errors from call_openrouter return {'error': 'exception', 'message': ...}
            if isinstance(result, dict) and result.get("error") == "exception":
                attempt += 1
                if attempt <= MAX_RETRIES:
                    backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"Exception for {run_id}; retrying after {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"❌ {run_id} failed after {MAX_RETRIES} retries: {result}")
                    break
            # otherwise assume success or non-retriable error; proceed to write output
            break

        output = {
            "run_id": run_id,
            "model_id": model_id,
            "order": order,
            "question_id": session.get("question_id"),
            "task": session.get("task"),
            "persona": session.get("persona"),
            "messages": messages,
            "response": result,
            "latency_sec": latency,
        }

        # Save raw output
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(output, f_out, indent=2, ensure_ascii=False)

        # Save chat history only
        hist_path = os.path.join(hist_dir, f"{run_id}.json")
        with open(hist_path, "w", encoding="utf-8") as f_hist:
            json.dump(messages, f_hist, indent=2, ensure_ascii=False)

        if latency is not None:
            print(f"✅ {run_id} done (latency: {latency:.2f}s)")
        else:
            print(f"❌ {run_id} failed: {result}")

        # delay between requests (configurable)
        time.sleep(REQUEST_DELAY)
