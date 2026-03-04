import json
import requests
from pathlib import Path

keys = json.load(open('config/model_keys.json', encoding='utf-8'))
sample = json.load(open('result/google_gemma-3-27b-it/woman_implicit_3/gsm8k_00563.json', encoding='utf-8'))
messages = sample['request']['messages']

candidates = [
    {'name': 'moonshot-direct', 'provider': 'moonshot', 'model_id': 'moonshot-v1-8k'},
    {'name': 'openrouter-polaris-alpha', 'provider': 'openrouter', 'model_id': 'openrouter/polaris-alpha'},
    {'name': 'openrouter-qwen3-4b-free', 'provider': 'openrouter', 'model_id': 'qwen/qwen3-4b:free'},
    {'name': 'openrouter-deepseek-r1-free', 'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1:free'},
    {'name': 'openrouter-longcat-free', 'provider': 'openrouter', 'model_id': 'meituan/longcat-flash-chat:free'},
    {'name': 'openrouter-glm45-air-free', 'provider': 'openrouter', 'model_id': 'z-ai/glm-4.5-air:free'},
    {'name': 'openrouter-grok41-fast-free', 'provider': 'openrouter', 'model_id': 'x-ai/grok-4.1-fast:free'},
    {'name': 'openrouter-bert-nebulon', 'provider': 'openrouter', 'model_id': 'openrouter/bert-nebulon-alpha'},
    {'name': 'openrouter-gemini25', 'provider': 'openrouter', 'model_id': 'google/gemini-2.5-flash'},
    {'name': 'openrouter-gemma27b', 'provider': 'openrouter', 'model_id': 'google/gemma-3-27b-it'},
    {'name': 'openrouter-gemma27b-free', 'provider': 'openrouter', 'model_id': 'google/gemma-3-27b-it:free'},
    {'name': 'openrouter-llama31-8b', 'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.1-8b-instruct'},
    {'name': 'openrouter-ministral8b', 'provider': 'openrouter', 'model_id': 'mistralai/ministral-8b-2512'},
    {'name': 'openrouter-gemma2-9b', 'provider': 'openrouter', 'model_id': 'google/gemma-2-9b-it'},
]

# index keys by (provider/model)
for cand in candidates:
    provider = cand['provider']
    model_id = cand['model_id']
    entry = None
    if provider == 'moonshot':
        for k in keys:
            if k.get('provider') == 'moonshot' and k.get('model_id') == model_id:
                entry = k
                break
    else:
        for k in keys:
            if k.get('model_id') == model_id:
                entry = k
                break
    if not entry:
        print(f"[NO_KEY] {cand['name']} model={model_id}")
        continue

    api_key = entry['api_key_alias']
    if provider == 'moonshot':
        url = entry.get('endpoint_url', 'https://api.moonshot.ai/v1/chat/completions')
    else:
        url = 'https://openrouter.ai/api/v1/chat/completions'

    payload = {
        'model': model_id,
        'messages': messages,
        'max_tokens': 1024,
        'temperature': 0.2,
    }

    try:
        resp = requests.post(
            url,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=120,
        )
        status = resp.status_code
        data = resp.json()
        choices = data.get('choices') if isinstance(data, dict) else None
        content = ''
        if choices:
            content = choices[0].get('message', {}).get('content', '') or ''
        ok = bool(content)
        err_msg = ''
        if not ok:
            if isinstance(data, dict):
                err = data.get('error')
                if isinstance(err, dict):
                    err_msg = err.get('message', str(err))
                else:
                    err_msg = str(err)
        print(f"[{ 'OK' if ok else 'FAIL' }] {cand['name']} status={status} len={len(content)}")
        if ok:
            print(' preview:', content[:180].replace('\n', ' '))
        else:
            print(' error :', (err_msg or 'no content')[:220])
    except Exception as e:
        print(f"[EXC] {cand['name']} -> {e}")
