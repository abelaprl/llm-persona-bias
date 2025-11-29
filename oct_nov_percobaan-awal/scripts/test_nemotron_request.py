import os
import json
import requests

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")
TASK_NAME = "mmlu_women_implicit_strong"
INPUT_DIR = os.path.join(BASE_DIR, "inputs", TASK_NAME, "14")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    models = json.load(f)

entry = None
for m in models:
    if 'nemotron' in m.get('model_id', '').lower() or 'nemotron' in m.get('name','').lower():
        entry = m
        break

if not entry:
    print('No nemotron entry found in config')
    raise SystemExit(1)

api_key = entry['api_key_alias']
model_id = entry['model_id']

# pick first input
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
if not files:
    print('No input files in', INPUT_DIR)
    raise SystemExit(1)

with open(os.path.join(INPUT_DIR, files[0]), 'r', encoding='utf-8') as f:
    session = json.load(f)

messages = session['messages']

payload = {
    'model': model_id,
    'messages': messages,
    'temperature': 0.2,
    'top_p': 1.0,
    'max_tokens': 1500,
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}

print('Sending test request to', model_id)
try:
    r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    print('Status:', r.status_code)
    try:
        print('Response JSON:', json.dumps(r.json(), indent=2))
    except Exception:
        print('Response text:', r.text)
except Exception as e:
    print('Request exception:', e)
