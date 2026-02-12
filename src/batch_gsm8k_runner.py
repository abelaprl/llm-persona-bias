"""
Batch GSM8K Experiment Runner dengan Retry Mechanism
Usage: python src/batch_gsm8k_runner.py --persona_id 1 --model_order 31-42 --questions 1-10
"""
import requests
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

def parse_range(range_str):
    """Parse range string like '1-10' or '31,32,35' to list"""
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]

def load_model_config(order):
    """Load specific model configuration"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    for m in models:
        if m['order'] == order:
            return m
    return None

def load_gsm8k_question(question_id):
    """Load specific GSM8K question"""
    with open('data/gsm8k_test.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    for item in dataset:
        if item.get('idx') == question_id or item.get('id') == question_id:
            return item
    
    if question_id <= len(dataset):
        return dataset[question_id - 1]
    
    return None

def load_warmup_result(model_id, persona_type, persona_id):
    """Load warmup conversation from file"""
    model_name = model_id.replace('/', '_')
    warmup_path = Path(f"result/{model_name}/{persona_type}_implicit_{persona_id}/warmup/persona_warmup.json")
    
    if not warmup_path.exists():
        return None
    
    with open(warmup_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_messages_array(warmup_data, gsm8k_question):
    """Build messages array with warmup + GSM8K"""
    warmup_q = warmup_data['input']['prompt']
    warmup_a = warmup_data['response']['content']
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": warmup_q},
        {"role": "assistant", "content": warmup_a},
        {"role": "user", "content": f"Solve the following math problem:\n\n{gsm8k_question}\n\nAnswer:"}
    ]
    
    return messages

def call_openrouter_api(model_config, messages, retry_count=0, max_retries=3):
    """Call OpenRouter API with retry mechanism"""
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
                "error": None,
                "retry_count": retry_count
            }
        elif response.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            # Rate limit or server error - retry dengan exponential backoff
            wait_time = 3 * (2 ** retry_count)  # 3s, 6s, 12s
            print(f"      [RETRY] Status {response.status_code} - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": latency,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "retry_count": retry_count
            }
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait_time = 5 * (2 ** retry_count)  # 5s, 10s, 20s
            print(f"      [RETRY] Timeout - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": time.time() - start_time,
                "error": "Timeout after retries",
                "retry_count": retry_count
            }
    except Exception as e:
        if retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"      [RETRY] Exception - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_config, messages, retry_count + 1, max_retries)
        else:
            return {
                "success": False,
                "api_response": None,
                "latency": time.time() - start_time,
                "error": str(e),
                "retry_count": retry_count
            }

def sanitize_model_name(model_id):
    """Convert model_id to folder name"""
    return model_id.replace('/', '_')

def create_result_json(model_config, warmup_data, gsm8k_data, messages, api_result, persona_type, persona_id):
    """Create result JSON structure"""
    model_name = sanitize_model_name(model_config['model_id'])
    timestamp = datetime.now().isoformat()
    
    result = {
        "metadata": {
            "run_id": f"{model_name}_{persona_type}_implicit_{persona_id}_gsm8k_{gsm8k_data.get('idx', gsm8k_data.get('id', 0)):05d}",
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
                "type": f"{persona_type}_implicit",
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
                "retry_count": api_result['retry_count']
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
                "retry_count": api_result['retry_count']
            }
        }
    
    result["analysis"] = {
        "extracted_answer": None,
        "is_correct": None
    }
    
    return result

def save_result(model_config, persona_type, persona_id, gsm8k_id, result_json):
    """Save result to file"""
    model_name = sanitize_model_name(model_config['model_id'])
    folder_path = Path(f"result/{model_name}/{persona_type}_implicit_{persona_id}")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    file_path = folder_path / f"gsm8k_{gsm8k_id:05d}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def check_result_exists(model_config, persona_type, persona_id, gsm8k_id):
    """Check if result file already exists"""
    model_name = sanitize_model_name(model_config['model_id'])
    file_path = Path(f"result/{model_name}/{persona_type}_implicit_{persona_id}/gsm8k_{gsm8k_id:05d}.json")
    return file_path.exists(), str(file_path)

def process_single_question(model_config, persona_type, persona_id, gsm8k_id, warmup_data):
    """Process single GSM8K question"""
    # Check if already processed
    exists, file_path = check_result_exists(model_config, persona_type, persona_id, gsm8k_id)
    if exists:
        return {
            "success": True,
            "skipped": True,
            "file_path": file_path,
            "latency": 0,
            "tokens": 0,
            "retry_count": 0
        }
    
    # Load GSM8K question
    gsm8k_data = load_gsm8k_question(gsm8k_id)
    if not gsm8k_data:
        return {"success": False, "skipped": False, "error": f"GSM8K question {gsm8k_id} not found"}
    
    # Build messages
    messages = build_messages_array(warmup_data, gsm8k_data['question'])
    
    # Call API with retry
    api_result = call_openrouter_api(model_config, messages)
    
    # Create and save result
    result_json = create_result_json(model_config, warmup_data, gsm8k_data, messages, api_result, persona_type, persona_id)
    file_path = save_result(model_config, persona_type, persona_id, gsm8k_id, result_json)
    
    return {
        "success": api_result['success'],
        "skipped": False,
        "file_path": file_path,
        "latency": api_result['latency'],
        "tokens": api_result['api_response'].get('usage', {}).get('total_tokens', 0) if api_result['success'] else 0,
        "retry_count": api_result['retry_count'],
        "error": api_result.get('error')
    }

def main():
    parser = argparse.ArgumentParser(description='Batch GSM8K Runner with Retry')
    parser.add_argument('--persona_id', type=int, required=True, help='Persona ID (1-3)')
    parser.add_argument('--persona_type', type=str, choices=['woman', 'man'], required=True, help='Persona type: woman or man')
    parser.add_argument('--model_order', type=str, required=True, help='Model order: single (31), range (31-35), or list (31,32,35)')
    parser.add_argument('--questions', type=str, required=True, help='Question range: single (1), range (1-10), or list (1,2,5)')
    
    args = parser.parse_args()
    
    # Parse inputs
    model_orders = parse_range(args.model_order)
    question_ids = parse_range(args.questions)
    
    print("="*80)
    print("BATCH GSM8K EXPERIMENT RUNNER")
    print("="*80)
    print(f"Persona: {args.persona_type}_implicit_{args.persona_id}")
    print(f"Models: {model_orders} ({len(model_orders)} models)")
    print(f"Questions: {question_ids[0]}-{question_ids[-1]} ({len(question_ids)} questions)")
    print(f"Total tasks: {len(model_orders) * len(question_ids)}")
    print("="*80)
    
    overall_start = time.time()
    all_results = []
    
    total_tasks = len(model_orders) * len(question_ids)
    current_task = 0
    
    for model_order in model_orders:
        # Load model config
        model_config = load_model_config(model_order)
        if not model_config:
            print(f"\n[ERROR] Model order {model_order} not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"MODEL: [{model_order}] {model_config['name']}")
        print(f"       {model_config['model_id']}")
        print(f"{'='*80}")
        
        # Load warmup once per model
        warmup_data = load_warmup_result(model_config['model_id'], args.persona_type, args.persona_id)
        if not warmup_data:
            print(f"[ERROR] Warmup not found for persona {args.persona_id}")
            continue
        
        print(f"[OK] Warmup loaded: {args.persona_type}_implicit_{args.persona_id}")
        
        model_start = time.time()
        model_success = 0
        model_failed = 0
        model_skipped = 0
        
        for gsm8k_id in question_ids:
            current_task += 1
            progress = f"{current_task}/{total_tasks}"
            
            print(f"\n  [{progress}] Question {gsm8k_id:05d}...", end=' ')
            
            result = process_single_question(model_config, args.persona_type, args.persona_id, gsm8k_id, warmup_data)
            
            if result.get('skipped', False):
                print(f"[SKIP] Already exists")
                model_skipped += 1
                model_success += 1  # Count as success for summary
            elif result['success']:
                print(f"[OK] {result['latency']:.2f}s | {result['tokens']} tokens", end='')
                if result['retry_count'] > 0:
                    print(f" | retries:{result['retry_count']}", end='')
                print()
                model_success += 1
            else:
                print(f"[FAIL] {result.get('error', 'Unknown error')[:50]}")
                model_failed += 1
            
            all_results.append({
                "model_order": model_order,
                "model_name": model_config['name'],
                "persona_type": args.persona_type,
                "persona_id": args.persona_id,
                "question_id": gsm8k_id,
                "status": "SKIPPED" if result.get('skipped', False) else ("SUCCESS" if result['success'] else "FAILED"),
                "latency": result.get('latency', 0),
                "tokens": result.get('tokens', 0),
                "retry_count": result.get('retry_count', 0),
                "error": result.get('error')
            })
            
            # Rate limiting (skip if file was skipped)
            if not result.get('skipped', False):
                time.sleep(1)
        
        model_elapsed = time.time() - model_start
        print(f"\n  Model summary: {model_success} success ({model_skipped} skipped), {model_failed} failed | {model_elapsed:.1f}s")
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    success_count = sum(1 for r in all_results if r['status'] in ['SUCCESS', 'SKIPPED'])
    skipped_count = sum(1 for r in all_results if r['status'] == 'SKIPPED')
    failed_count = sum(1 for r in all_results if r['status'] == 'FAILED')
    total_tokens = sum(r['tokens'] for r in all_results)
    
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(all_results)}")
    print(f"Success: {success_count} ({skipped_count} skipped, {success_count - skipped_count} new)")
    print(f"Failed: {failed_count}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"Avg time/task: {overall_elapsed/len(all_results):.1f}s")
    
    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"logs_batch_persona{args.persona_id}_models{model_orders[0]}-{model_orders[-1]}_q{question_ids[0]}-{question_ids[-1]}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "persona_id": args.persona_id,
                "model_orders": model_orders,
                "question_ids": question_ids
            },
            "summary": {
                "total_tasks": len(all_results),
                "success": success_count,
                "skipped": skipped_count,
                "new": success_count - skipped_count,
                "failed": failed_count,
                "total_tokens": total_tokens,
                "elapsed_seconds": overall_elapsed
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nLog saved: {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
