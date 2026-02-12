import requests
import json
import time
from datetime import datetime
from pathlib import Path

# Load model
with open('config/model_keys.json', 'r') as f:
    models = json.load(f)
model = [m for m in models if m['order'] == 42][0]

# Load prompts
with open('inputs/implicits_woman_prompts.json', 'r') as f:
    prompts = json.load(f)

print('=' * 60)
print('WARMUP TEST: Order 42 - allenai/olmo-3-32b-think')
print('=' * 60)
print(f"Model: {model['name']}")
print(f"Model ID: {model['model_id']}")

results = []
for prompt in prompts[:3]:
    print(f"\nTesting prompt {prompt['id']}...")
    
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': f"Bearer {model['api_key_alias']}",
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model['model_id'],
        'messages': [
            {'role': 'system', 'content': 'You are a helpful AI assistant.'},
            {'role': 'user', 'content': prompt['prompt']}
        ],
        'temperature': 0.2,
        'max_tokens': 2048
    }
    
    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)  # 5 min timeout for thinking model
        latency = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get('usage', {}).get('total_tokens', 0)
            print(f'  ✅ SUCCESS - {tokens} tokens, {latency:.2f}s')
            
            # Save
            model_folder = model['model_id'].replace('/', '_')
            out_dir = Path(f"result/{model_folder}/woman_implicit_{prompt['id']}/warmup")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir / 'persona_warmup.json', 'w') as f:
                json.dump({
                    'metadata': {'model_id': model['model_id'], 'timestamp': datetime.now().isoformat()},
                    'input': {'prompt_id': prompt['id'], 'prompt': prompt['prompt']},
                    'response': {'success': True, 'content': data['choices'][0]['message']['content'], 'usage': data.get('usage', {}), 'latency': latency}
                }, f, indent=2)
            
            results.append({'order': 42, 'prompt_id': prompt['id'], 'status': 'SUCCESS', 'tokens': tokens, 'latency': latency})
        else:
            print(f'  ❌ FAIL - HTTP {resp.status_code}: {resp.text[:200]}')
            results.append({'order': 42, 'prompt_id': prompt['id'], 'status': 'FAILED', 'error': resp.text[:200]})
    except Exception as e:
        print(f'  ❌ FAIL - {str(e)}')
        results.append({'order': 42, 'prompt_id': prompt['id'], 'status': 'FAILED', 'error': str(e)})
    
    time.sleep(2)

# Save log
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'logs_warmup_order42_{ts}.json', 'w') as f:
    json.dump(results, f, indent=2)

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f'\n✅ Log saved')
print(f'Summary: {success_count}/{len(results)} successful')
