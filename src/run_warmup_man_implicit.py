"""
WARMUP PHASE: Man Implicit Prompts (Models 31-49)
Load man implicit prompts dan jalankan warmup untuk semua model

Usage: python src/run_warmup_man_implicit.py
"""
import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path

def load_config():
    """Load model configuration"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    return [m for m in models if 31 <= m['order'] <= 49]

def load_prompts():
    """Load man implicit prompts"""
    with open('inputs/implicits_man_prompts.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def sanitize_model_name(model_id):
    """Convert model_id to folder name: openai/gpt-oss-20b -> openai_gpt-oss-20b"""
    return model_id.replace('/', '_')

def call_openrouter_api(model_config, prompt_text, prompt_id):
    """Call OpenRouter API with standard system prompt"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_config['api_key_alias']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt_text}
    ]
    
    payload = {
        "model": model_config['model_id'],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            api_response = response.json()
            return {
                "success": True,
                "api_response": api_response,
                "latency": latency,
                "error": None
            }
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "api_response": None,
            "latency": latency,
            "error": str(e)
        }

def create_warmup_json(model_config, prompt_data, api_result):
    """Create JSON structure matching existing format"""
    model_name = sanitize_model_name(model_config['model_id'])
    timestamp = datetime.now().isoformat()
    
    result = {
        "metadata": {
            "run_id": f"{model_name}_man_implicit_{prompt_data['id']}_warmup",
            "experiment_type": "warmup_persona",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_config['model_id'],
                "model_name": model_config['name'],
                "api_endpoint": "https://openrouter.ai/api/v1/chat/completions"
            }
        },
        "input": {
            "prompt_id": prompt_data['id'],
            "prompt_type": "man_implicit",
            "prompt": prompt_data['prompt']
        },
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_data['prompt']}
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_config['model_id']
            }
        }
    }
    
    if api_result['success']:
        api_resp = api_result['api_response']
        choice = api_resp['choices'][0]
        message = choice['message']
        
        result["response"] = {
            "success": True,
            "content": message.get('content', ''),
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result['latency'],
                "tokens": api_resp.get('usage', {}),
                "retry_count": 0
            }
        }
    else:
        result["response"] = {
            "success": False,
            "content": None,
            "raw_api_response": None,
            "error": api_result['error'],
            "performance": {
                "latency_sec": api_result['latency'],
                "tokens": {},
                "retry_count": 0
            }
        }
    
    return result

def save_warmup_result(model_config, prompt_id, warmup_json):
    """Save warmup result to appropriate folder"""
    model_name = sanitize_model_name(model_config['model_id'])
    
    # Create folder structure: result/{model_name}/man_implicit_{id}/warmup/
    folder_path = Path(f"result/{model_name}/man_implicit_{prompt_id}/warmup")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save to persona_warmup.json
    file_path = folder_path / "persona_warmup.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(warmup_json, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def main():
    print("="*80)
    print("WARMUP PHASE: Man Implicit Prompts (Models 31-49)")
    print("="*80)
    
    # Load configuration
    models = load_config()
    prompts = load_prompts()
    
    print(f"\nLoaded {len(models)} models (order 31-49)")
    print(f"Loaded {len(prompts)} prompts (man implicit)\n")
    
    total_tasks = len(models) * len(prompts)
    current_task = 0
    
    results_summary = []
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"[{model['order']}] {model['name']}")
        print(f"    Model ID: {model['model_id']}")
        print(f"{'='*80}")
        
        for prompt_data in prompts:
            current_task += 1
            print(f"\n  [{current_task}/{total_tasks}] Processing prompt ID {prompt_data['id']}...")
            prompt_preview = prompt_data['prompt'][:80].encode('ascii', errors='ignore').decode('ascii')
            print(f"    Prompt preview: {prompt_preview}...")
            
            # Call API
            api_result = call_openrouter_api(model, prompt_data['prompt'], prompt_data['id'])
            
            # Create JSON structure
            warmup_json = create_warmup_json(model, prompt_data, api_result)
            
            # Save result
            try:
                file_path = save_warmup_result(model, prompt_data['id'], warmup_json)
                
                if api_result['success']:
                    usage = api_result['api_response'].get('usage', {})
                    print(f"    [OK] SUCCESS - Saved to: {file_path}")
                    print(f"      Tokens: {usage.get('total_tokens', 'N/A')} | Latency: {api_result['latency']:.2f}s")
                    
                    results_summary.append({
                        "model_order": model['order'],
                        "model_name": model['name'],
                        "prompt_id": prompt_data['id'],
                        "status": "SUCCESS",
                        "file_path": file_path,
                        "tokens": usage.get('total_tokens', 0),
                        "latency": api_result['latency']
                    })
                else:
                    print(f"    [FAIL] FAILED - {api_result['error'][:100]}")
                    print(f"      Saved error log to: {file_path}")
                    
                    results_summary.append({
                        "model_order": model['order'],
                        "model_name": model['name'],
                        "prompt_id": prompt_data['id'],
                        "status": "FAILED",
                        "error": api_result['error'][:100],
                        "file_path": file_path
                    })
            
            except Exception as e:
                print(f"    [ERROR] EXCEPTION - Failed to save: {str(e)}")
                results_summary.append({
                    "model_order": model['order'],
                    "model_name": model['name'],
                    "prompt_id": prompt_data['id'],
                    "status": "ERROR",
                    "error": str(e)
                })
            
            # Rate limiting protection
            time.sleep(1)
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results_summary if r['status'] in ['FAILED', 'ERROR'])
    
    print(f"\n[OK] SUCCESS: {success_count}/{total_tasks} tasks")
    print(f"[FAIL] FAILED:  {failed_count}/{total_tasks} tasks")
    
    if failed_count > 0:
        print("\nFailed tasks:")
        for r in results_summary:
            if r['status'] in ['FAILED', 'ERROR']:
                print(f"  [{r['model_order']}] {r['model_name']} - Prompt {r['prompt_id']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = f"logs_warmup_man_implicit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "="*80)
    print("WARMUP PHASE COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
