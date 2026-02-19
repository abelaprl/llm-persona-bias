import requests
import json
import time

# Load config
with open('config/model_keys.json', 'r') as f:
    models = json.load(f)

# Get models order 37-49
test_models = [m for m in models if 37 <= m['order'] <= 49]

print(f"Testing {len(test_models)} models (order 37-49)...\n")
print("="*80)

results = []

for model in test_models:
    print(f"\n[{model['order']}] {model['name']}")
    print(f"    Model ID: {model['model_id']}")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model['api_key_alias']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model['model_id'],
        "messages": [
            {"role": "user", "content": "Say OK"}
        ],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ SUCCESS - Response: {result['choices'][0]['message']['content'][:50]}")
            results.append({"order": model['order'], "name": model['name'], "status": "OK"})
        else:
            error_msg = response.json().get('error', {}).get('message', response.text)[:100]
            print(f"    ✗ FAILED - Status {response.status_code}: {error_msg}")
            results.append({"order": model['order'], "name": model['name'], "status": f"FAIL: {error_msg}"})
            
    except Exception as e:
        print(f"    ✗ EXCEPTION - {str(e)[:100]}")
        results.append({"order": model['order'], "name": model['name'], "status": f"ERROR: {str(e)[:50]}"})
    
    time.sleep(0.5)  # Rate limit protection

print("\n" + "="*80)
print("\nSUMMARY:")
print("="*80)

success = [r for r in results if r['status'] == 'OK']
failed = [r for r in results if r['status'] != 'OK']

print(f"\n✓ SUCCESS: {len(success)}/{len(results)} models")
for r in success:
    print(f"  [{r['order']}] {r['name']}")

if failed:
    print(f"\n✗ FAILED: {len(failed)}/{len(results)} models")
    for r in failed:
        print(f"  [{r['order']}] {r['name']}: {r['status']}")

# Save results
with open('test_models_37_49_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: test_models_37_49_results.json")
