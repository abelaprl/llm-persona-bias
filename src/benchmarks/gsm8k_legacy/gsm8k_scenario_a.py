"""
SCENARIO A: Independent Evaluation per GSM8K item
Menggunakan warmup conversation dari file JSON yang sudah ada (tidak panggil API warmup lagi)

Usage: python src/benchmarks/gsm8k_legacy/gsm8k_scenario_a.py --model_order 31 --gsm8k_id 1 --persona_id 1
"""
import requests
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

def load_model_config(order):
    """Load specific model configuration"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    for m in models:
        if m['order'] == order:
            return m
    raise ValueError(f"Model order {order} not found")

def load_gsm8k_question(question_id):
    """Load specific GSM8K question"""
    with open('data/gsm8k_test.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    for item in dataset:
        if item.get('idx') == question_id or item.get('id') == question_id:
            return item
    
    # Fallback: use index
    if question_id <= len(dataset):
        return dataset[question_id - 1]
    
    raise ValueError(f"GSM8K question {question_id} not found")

def load_warmup_result(model_id, persona_id):
    """Load warmup conversation result (sudah ada di file, tidak perlu API call)"""
    model_name = model_id.replace('/', '_')
    warmup_path = Path(f"result/{model_name}/woman_implicit_{persona_id}/warmup/persona_warmup.json")
    
    if not warmup_path.exists():
        raise FileNotFoundError(f"Warmup result not found: {warmup_path}")
    
    with open(warmup_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sanitize_model_name(model_id):
    """Convert model_id to folder name"""
    return model_id.replace('/', '_')

def build_messages_array(warmup_data, gsm8k_question):
    """Build messages array with warmup conversation + GSM8K question
    
    Conversation structure:
    1. System: "You are a helpful AI assistant."
    2. User: {warmup question} (load dari file JSON warmup)
    3. Assistant: {warmup response} (load dari file JSON warmup - TIDAK PANGGIL API)
    4. User: "Solve the following math problem:\n\n{gsm8k_question}\n\nAnswer:"
    
    Hanya 1 API call untuk jawaban GSM8K (message ke-4)
    """
    warmup_q = warmup_data['input']['prompt']
    warmup_a = warmup_data['response']['content']
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": warmup_q},
        {"role": "assistant", "content": warmup_a},
        {"role": "user", "content": f"Solve the following math problem:\n\n{gsm8k_question}\n\nAnswer:"}
    ]
    
    return messages

def call_openrouter_api(model_config, messages):
    """Call OpenRouter API dengan full conversation history"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_config['api_key_alias']}",
        "Content-Type": "application/json"
    }
    
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

def create_result_json(model_config, warmup_data, gsm8k_data, messages, api_result, persona_id):
    """Create result JSON structure"""
    model_name = sanitize_model_name(model_config['model_id'])
    timestamp = datetime.now().isoformat()
    
    result = {
        "metadata": {
            "run_id": f"{model_name}_woman_implicit_{persona_id}_gsm8k_{gsm8k_data.get('idx', gsm8k_data.get('id', 0)):05d}",
            "experiment_type": "gsm8k_bias_study",
            "scenario": "A",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_config['model_id'],
                "model_name": model_config['name'],
                "api_endpoint": "https://openrouter.ai/api/v1/chat/completions"
            }
        },
        "input": {
            "question_id": gsm8k_data.get('idx', gsm8k_data.get('id', 0)),
            "dataset": "gsm8k_test",
            "question": gsm8k_data['question'],
            "ground_truth": gsm8k_data.get('answer', ''),
            "persona": {
                "id": persona_id,
                "type": "woman_implicit",
                "warmup_prompt": warmup_data['input']['prompt'],
                "warmup_response": warmup_data['response']['content']
            }
        },
        "request": {
            "messages": messages,
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
    
    result["analysis"] = {
        "extracted_answer": None,
        "is_correct": None
    }
    
    return result

def save_result(model_config, persona_id, gsm8k_id, result_json):
    """Save result to appropriate folder"""
    model_name = sanitize_model_name(model_config['model_id'])
    
    # Create folder: result/{model_name}/woman_implicit_{persona_id}/
    folder_path = Path(f"result/{model_name}/woman_implicit_{persona_id}")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save to gsm8k_{id:05d}.json
    file_path = folder_path / f"gsm8k_{gsm8k_id:05d}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def main():
    parser = argparse.ArgumentParser(description='Run GSM8K experiment (Scenario A - Independent Evaluation)')
    parser.add_argument('--model_order', type=int, required=True, help='Model order (31-49)')
    parser.add_argument('--gsm8k_id', type=int, required=True, help='GSM8K question ID')
    parser.add_argument('--persona_id', type=int, required=True, help='Persona ID (1-3)')
    
    args = parser.parse_args()
    
    print(f"="*80)
    print(f"SCENARIO A: Independent Evaluation")
    print(f"Model Order: {args.model_order} | GSM8K ID: {args.gsm8k_id} | Persona ID: {args.persona_id}")
    print(f"="*80)
    
    try:
        # Load configuration
        model_config = load_model_config(args.model_order)
        print(f"\nModel: [{args.model_order}] {model_config['name']}")
        print(f"       {model_config['model_id']}")
        
        # Load warmup result (dari file JSON - TIDAK panggil API lagi)
        warmup_data = load_warmup_result(model_config['model_id'], args.persona_id)
        print(f"\n[OK] Warmup loaded from file (no API call)")
        warmup_q_preview = warmup_data['input']['prompt'][:80].encode('ascii', errors='ignore').decode('ascii')
        warmup_a_preview = warmup_data['response']['content'][:80].encode('ascii', errors='ignore').decode('ascii')
        print(f"  Warmup Q: {warmup_q_preview}...")
        print(f"  Warmup A: {warmup_a_preview}...")
        
        # Load GSM8K question
        gsm8k_data = load_gsm8k_question(args.gsm8k_id)
        print(f"\n[OK] GSM8K loaded: Question {args.gsm8k_id}")
        print(f"  Question: {gsm8k_data['question'][:80]}...")
        
        # Build messages array (conversation history)
        messages = build_messages_array(warmup_data, gsm8k_data['question'])
        print(f"\n[OK] Messages array built: {len(messages)} messages")
        print(f"  [1] System: You are a helpful AI assistant.")
        print(f"  [2] User: {messages[1]['content'][:60]}...")
        print(f"  [3] Assistant: {messages[2]['content'][:60]}... (from file)")
        print(f"  [4] User: Solve the following math problem...")
        
        # Call API HANYA SEKALI untuk jawaban GSM8K
        print(f"\n[CALLING API] GSM8K answer (only 1 API call)...")
        api_result = call_openrouter_api(model_config, messages)
        
        if api_result['success']:
            usage = api_result['api_response'].get('usage', {})
            answer = api_result['api_response']['choices'][0]['message']['content']
            answer_preview = answer[:150].encode('ascii', errors='ignore').decode('ascii')
            print(f"\n[SUCCESS] API call SUCCESS")
            print(f"  Tokens: {usage.get('total_tokens', 'N/A')} | Latency: {api_result['latency']:.2f}s")
            print(f"  Answer: {answer_preview}...")
            
            # Create and save result
            result_json = create_result_json(model_config, warmup_data, gsm8k_data, messages, api_result, args.persona_id)
            file_path = save_result(model_config, args.persona_id, args.gsm8k_id, result_json)
            
            print(f"\n[SAVED] {file_path}")
        else:
            print(f"\n[FAILED] API call FAILED: {api_result['error']}")
            
            # Still save error result
            result_json = create_result_json(model_config, warmup_data, gsm8k_data, messages, api_result, args.persona_id)
            file_path = save_result(model_config, args.persona_id, args.gsm8k_id, result_json)
            print(f"[SAVED] Error log saved to: {file_path}")
        
        print(f"\n{'='*80}")
        print(f"COMPLETED")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n[ERROR] EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
