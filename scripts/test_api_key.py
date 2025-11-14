#!/usr/bin/env python3
"""
Quick test: check if API key works for model polaris-alpha
"""
import json
import requests
from pathlib import Path

base = Path(__file__).parent.parent
with open(base / 'config' / 'model_keys.json', 'r') as f:
    models = json.load(f)

# Test model 1 (polaris-alpha)
model = models[0]
print(f"Testing Model {model['order']}: {model['model_id']}")
print(f"API Key: {model['api_key_alias'][:20]}...")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {model['api_key_alias']}",
    "Content-Type": "application/json"
}
payload = {
    "model": model['model_id'],
    "messages": [{"role": "user", "content": "Say hello"}]
}

response = requests.post(url, headers=headers, json=payload, timeout=30)
print(f"\nStatus: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✅ Success! Response: {data['choices'][0]['message']['content'][:100]}")
else:
    print(f"❌ Error: {response.text}")
