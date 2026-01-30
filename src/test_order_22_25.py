from src.load_keys import load_model_keys, get_models
import requests

models = get_models(load_model_keys())
for order in range(22, 26):
    m = next((x for x in models if x.get('order') == order), None)
    if m:
        print(f"\n[TEST] Order {order}: {m['name']}")
        print(f"       Model ID: {m['model_id']}")
        print(f"       Provider: {m.get('provider', 'openrouter')}")
        headers = {'Authorization': f"Bearer {m['api_key_alias']}", 'Content-Type': 'application/json'}
        payload = {'model': m['model_id'], 'messages': [{'role': 'user', 'content': 'What is 2 + 2? Answer briefly in one line.'}], 'max_tokens': 20}
        url = m.get('endpoint_url', 'https://openrouter.ai/api/v1/chat/completions')
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                print(' ... ✓ OK')
            else:
                print(f' ... ✗ FAIL\n       Error: HTTP {resp.status_code}: {resp.text[:200]}')
        except Exception as e:
            print(f' ... ✗ FAIL\n       Error: {str(e)[:200]}')
    else:
        print(f"Order {order}: NOT FOUND")
