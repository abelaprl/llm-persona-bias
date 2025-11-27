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
    if m.get('model_id') == 'x-ai/grok-4.1-fast':
        entry = m
        break

if not entry:
    print('No grok entry found in config')
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
    'reasoning': 'chain_of_thought',
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

    if r.status_code == 400:
        # Try alternative payload: some models expect `input` as a single string
        print('\nRetrying with single-string `input` payload...')
        # convert messages list to a single conversation string (last user content preferred)
        convo = []
        for m in messages:
            role = m.get('role', 'user')
            content = m.get('content', '')
            convo.append(f"[{role}] {content}")
        input_text = '\n'.join(convo)
        payload2 = {
            'model': model_id,
            'input': input_text,
            'temperature': 0.2,
            'max_tokens': 1500,
            'reasoning': 'chain_of_thought',
        }
        r2 = requests.post(API_URL, headers=headers, json=payload2, timeout=60)
        print('Status2:', r2.status_code)
        try:
            print('Response2 JSON:', json.dumps(r2.json(), indent=2))
        except Exception:
            print('Response2 text:', r2.text)
        if r2.status_code == 400:
            # Try final fallback: send a `prompt` with the last user content as plain string
            print('\nRetrying with `prompt` as last user message...')
            last_user = None
            for m in reversed(messages):
                if m.get('role') == 'user' or m.get('role') == 'assistant':
                    last_user = m.get('content')
                    break
            if not last_user:
                last_user = '\n'.join([m.get('content','') for m in messages])
            payload3 = {
                'model': model_id,
                'prompt': last_user,
                'temperature': 0.2,
                'max_tokens': 1500,
                'reasoning': 'chain_of_thought',
            }
            r3 = requests.post(API_URL, headers=headers, json=payload3, timeout=60)
            print('Status3:', r3.status_code)
            try:
                print('Response3 JSON:', json.dumps(r3.json(), indent=2))
            except Exception:
                print('Response3 text:', r3.text)
except Exception as e:
    print('Request exception:', e)
