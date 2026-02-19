import requests
import json

# Load config
with open('config/model_keys.json', 'r') as f:
    models = json.load(f)

# Get order 36
model_config = [m for m in models if m['order'] == 36][0]

print(f"Testing model: {model_config['name']}")
print(f"Model ID: {model_config['model_id']}")

# Test API call
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {model_config['api_key_alias']}",
    "Content-Type": "application/json"
}

payload = {
    "model": model_config['model_id'],
    "messages": [
        {"role": "user", "content": "Hello! Just testing. Reply with 'OK' if you receive this."}
    ],
    "max_tokens": 50
}

print("\nSending test request...")
try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ SUCCESS!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Model used: {result.get('model', 'N/A')}")
    else:
        print(f"\n✗ FAILED!")
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"\n✗ EXCEPTION!")
    print(f"Error: {str(e)}")
