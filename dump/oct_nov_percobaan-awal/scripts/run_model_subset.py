import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")
TASK_NAME = "mmlu_women_implicit_strong"
INPUT_ROOT = os.path.join(BASE_DIR, "inputs", TASK_NAME)
LOGS_ROOT = os.path.join(BASE_DIR, "logs_json", TASK_NAME)
HISTORY_ROOT = os.path.join(BASE_DIR, "chat_histories", TASK_NAME)

os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(HISTORY_ROOT, exist_ok=True)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_TIMEOUT = 60

# Runtime configuration via env
TARGET_MODEL_ID = os.environ.get("TARGET_MODEL_ID")
TARGET_ORDER = os.environ.get("TARGET_ORDER")
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "10"))
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
    if ENABLE_REASONING:
        payload["reasoning"] = "chain_of_thought"

    try:
        start = time.time()
        r = requests.post(API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        latency = time.time() - start
        try:
            return r.json(), latency, r.status_code
        except Exception:
            return {"error": r.status_code, "message": r.text}, latency, r.status_code
    except Exception as e:
        return {"error": "exception", "message": str(e)}, None, None


# Load models
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    models = json.load(f)

selected = None
for m in models:
    if TARGET_MODEL_ID and m.get('model_id') == TARGET_MODEL_ID:
        selected = m
        break
    if not TARGET_MODEL_ID and TARGET_ORDER:
        try:
            if int(m.get('order')) == int(TARGET_ORDER):
                selected = m
                break
        except Exception:
            pass

if not selected:
    print('No matching model found in config for TARGET_MODEL_ID/TARGET_ORDER')
    raise SystemExit(1)

model_id = selected['model_id']
api_key = selected['api_key_alias']
order = selected.get('order')

# Use TARGET_ORDER (env) for which inputs to run; fall back to model's configured order
use_order = TARGET_ORDER if TARGET_ORDER is not None else order
try:
    use_order_int = int(use_order)
except Exception:
    use_order_int = int(order)

input_dir = os.path.join(INPUT_ROOT, f"{use_order_int:02d}")
if not os.path.isdir(input_dir):
    print('Input dir not found:', input_dir)
    raise SystemExit(1)

log_dir = os.path.join(LOGS_ROOT, f"{int(order):02d}")
hist_dir = os.path.join(HISTORY_ROOT, f"{int(order):02d}")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)

print(f"Running subset {SUBSET_SIZE} for model {model_id} on inputs in {input_dir}")

files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
files = files[:SUBSET_SIZE]

for fname in files:
    session_path = os.path.join(input_dir, fname)
    with open(session_path, 'r', encoding='utf-8') as f:
        session = json.load(f)

    run_id = session['run_id']
    messages = session['messages']

    out_path = os.path.join(log_dir, f"{run_id}.json")
    hist_path = os.path.join(hist_dir, f"{run_id}.json")

    if os.path.exists(out_path):
        print('↪️  Skipping existing', run_id)
        if not os.path.exists(hist_path):
            with open(hist_path, 'w', encoding='utf-8') as fh:
                json.dump(messages, fh, indent=2, ensure_ascii=False)
        continue

    attempt = 0
    while True:
        result, latency, status = call_openrouter(model_id, api_key, messages)
        # handle rate limit
        if isinstance(result, dict) and result.get('error') == 429:
            attempt += 1
            if attempt <= MAX_RETRIES:
                backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"429 rate limit for {run_id}; retrying after {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            else:
                print(f"❌ {run_id} failed after {MAX_RETRIES} retries: {result}")
                break
        if isinstance(result, dict) and result.get('error') == 'exception':
            attempt += 1
            if attempt <= MAX_RETRIES:
                backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"Exception for {run_id}; retrying after {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            else:
                print(f"❌ {run_id} failed after {MAX_RETRIES} retries: {result}")
                break
        # success or non-retriable
        break

    output = {
        'run_id': run_id,
        'model_id': model_id,
        'order': order,
        'question_id': session.get('question_id'),
        'task': session.get('task'),
        'persona': session.get('persona'),
        'messages': messages,
        'response': result,
        'latency_sec': latency,
    }

    with open(out_path, 'w', encoding='utf-8') as fo:
        json.dump(output, fo, indent=2, ensure_ascii=False)

    with open(hist_path, 'w', encoding='utf-8') as fh:
        json.dump(messages, fh, indent=2, ensure_ascii=False)

    if latency is not None:
        print(f"✅ {run_id} done (latency: {latency:.2f}s)")
    else:
        print(f"❌ {run_id} failed: {result}")

    time.sleep(REQUEST_DELAY)

print('Subset run complete.')
