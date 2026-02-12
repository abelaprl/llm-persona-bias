import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path

def load_config():
    """Load model configuration - ONLY order 33"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    return [m for m in models if m['order'] == 33]

def load_prompts():
    """Load woman implicit prompts"""
    with open('inputs/implicits_woman_prompts.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def sanitize_model_name(model_id):
    """Convert model_id to folder name"""
    return model_id.replace('/', '_')

def call_openrouter_api(model_config, prompt_text, prompt_id):
    """Call OpenRouter API"""
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
        response = requests.post(url, headers=headers, json=payload, timeout=120)
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
    """Create JSON structure"""
    model_name = sanitize_model_name(model_config['model_id'])
    timestamp = datetime.now().isoformat()
    
    result = {
        "metadata": {
            "run_id": f"{model_name}_woman_implicit_{prompt_data['id']}_warmup",
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
            "prompt_type": "woman_implicit",
            "prompt": prompt_data['prompt']
        },
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_data['prompt']}
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048
            }
        },
        "response": {}
    }
    
    if api_result["success"]:
        api_resp = api_result["api_response"]
        result["response"] = {
            "success": True,
            "content": api_resp.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "usage": api_resp.get("usage", {}),
            "latency_seconds": api_result["latency"],
            "finish_reason": api_resp.get("choices", [{}])[0].get("finish_reason"),
            "model_used": api_resp.get("model")
        }
    else:
        result["response"] = {
            "success": False,
            "error": api_result["error"],
            "latency_seconds": api_result["latency"]
        }
    
    return result

def main():
    print("=" * 60)
    print("WARMUP TEST: Order 33 - qwen/qwen3-235b-a22b-2507")
    print("=" * 60)
    
    models = load_config()
    prompts = load_prompts()
    
    if not models:
        print("ERROR: Model order 33 tidak ditemukan di config!")
        return
    
    model = models[0]
    print(f"\nModel: {model['name']}")
    print(f"Model ID: {model['model_id']}")
    
    # Test with 3 prompts
    test_prompts = prompts[:3]
    results = []
    
    for prompt_data in test_prompts:
        print(f"\nTesting prompt {prompt_data['id']}...")
        
        api_result = call_openrouter_api(model, prompt_data['prompt'], prompt_data['id'])
        
        if api_result["success"]:
            tokens = api_result["api_response"].get("usage", {}).get("total_tokens", 0)
            print(f"  ✅ SUCCESS - {tokens} tokens, {api_result['latency']:.2f}s")
            
            # Save result
            model_folder = sanitize_model_name(model['model_id'])
            output_dir = Path(f"result/{model_folder}/woman_implicit_{prompt_data['id']}/warmup")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            warmup_json = create_warmup_json(model, prompt_data, api_result)
            output_file = output_dir / "persona_warmup.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(warmup_json, f, indent=2, ensure_ascii=False)
            
            results.append({
                "model_order": model['order'],
                "model_name": model['name'],
                "prompt_id": prompt_data['id'],
                "status": "SUCCESS",
                "file_path": str(output_file),
                "tokens": tokens,
                "latency": api_result['latency']
            })
        else:
            print(f"  ❌ FAILED - {api_result['error']}")
            results.append({
                "model_order": model['order'],
                "model_name": model['name'],
                "prompt_id": prompt_data['id'],
                "status": "FAILED",
                "error": api_result['error']
            })
        
        time.sleep(2)  # Rate limit
    
    # Save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs_warmup_order33_{timestamp}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Log saved to: {log_file}")
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\nSummary: {success_count}/{len(results)} successful")

if __name__ == "__main__":
    main()
