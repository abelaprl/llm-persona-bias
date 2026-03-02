"""
Batch GSM8K Experiment Runner untuk Explicit Prompts dengan Retry Mechanism
Usage: python src/batch_explicit_runner.py --persona_type man --persona_id 1 --model_order 31-42 --questions 1-1319
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

def load_explicit_prompt(persona_type, persona_id):
    """Load explicit prompt"""
    file_path = f'inputs/explicits_{persona_type}_prompts.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    for p in prompts:
        if p['id'] == persona_id:
            return p
    return None

def call_openrouter_api(model_config, user_message, retry_count=0):
    """Call OpenRouter API with retry mechanism"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_config['api_key_alias']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_message}
    ]
    
    payload = {
        "model": model_config['model_id'],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    max_retries = 3
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                api_response = response.json()
                return {
                    "success": True,
                    "response": api_response,
                    "latency": latency,
                    "retry_count": retry_count + attempt,
                    "error": None
                }
            elif response.status_code in [429, 500, 502, 503, 504]:
                # Rate limit or server error - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 3 * (2 ** attempt)  # 3s, 6s, 12s
                    print(f"      [RETRY] HTTP {response.status_code} - waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "response": None,
                        "latency": latency,
                        "retry_count": retry_count + attempt + 1,
                        "error": f"HTTP {response.status_code} after {max_retries} retries"
                    }
            else:
                # Other errors - don't retry
                return {
                    "success": False,
                    "response": None,
                    "latency": latency,
                    "retry_count": retry_count + attempt,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }
                
        except requests.exceptions.Timeout:
            # Timeout - retry with longer backoff
            latency = time.time() - start_time
            if attempt < max_retries - 1:
                wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"      [RETRY] Timeout - waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "response": None,
                    "latency": latency,
                    "retry_count": retry_count + attempt + 1,
                    "error": f"Timeout after {max_retries} retries"
                }
                
        except Exception as e:
            # Other exceptions - retry with backoff
            latency = time.time() - start_time
            if attempt < max_retries - 1:
                wait_time = 3 * (2 ** attempt)
                print(f"      [RETRY] Exception: {str(e)[:50]} - waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "response": None,
                    "latency": latency,
                    "retry_count": retry_count + attempt + 1,
                    "error": f"Exception after {max_retries} retries: {str(e)}"
                }
    
    # Should not reach here
    return {
        "success": False,
        "response": None,
        "latency": time.time() - start_time,
        "retry_count": retry_count + max_retries,
        "error": "Unknown error"
    }

def create_result_json(model_config, explicit_prompt, gsm8k_data, user_message, api_result, persona_type, persona_id):
    """Create result JSON matching implicit format"""
    model_name = model_config['model_id'].replace('/', '_')
    timestamp = datetime.now().isoformat()
    question_id = gsm8k_data.get('idx', gsm8k_data.get('id', 0))
    
    content = ""
    if api_result['success']:
        content = api_result['response']['choices'][0]['message'].get('content', '')
    
    result = {
        "metadata": {
            "run_id": f"{model_name}_{persona_type}_explicit_{persona_id}_gsm8k_{question_id:05d}",
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
            "question_id": question_id,
            "dataset": "gsm8k_test",
            "question": gsm8k_data['question'],
            "ground_truth": gsm8k_data.get('answer', ''),
            "persona": {
                "id": persona_id,
                "type": f"{persona_type}_explicit",
                "explicit_prompt": explicit_prompt['prompt']
            }
        },
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 2048,
                "model": model_config['model_id']
            }
        },
        "response": {
            "success": api_result['success'],
            "content": content,
            "raw_api_response": api_result.get('response'),
            "latency": api_result['latency']
        },
        "retry_count": api_result.get('retry_count', 0)
    }
    
    if not api_result['success']:
        result['response']['error'] = api_result.get('error')
    
    return result

def save_result(model_config, persona_type, persona_id, gsm8k_id, result_json):
    """Save result to file"""
    model_name = model_config['model_id'].replace('/', '_')
    folder_path = Path(f"result/{model_name}/{persona_type}_explicit_{persona_id}")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    file_path = folder_path / f"gsm8k_{gsm8k_id:05d}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def check_result_exists(model_config, persona_type, persona_id, gsm8k_id):
    """Check if result file already exists"""
    model_name = model_config['model_id'].replace('/', '_')
    file_path = Path(f"result/{model_name}/{persona_type}_explicit_{persona_id}/gsm8k_{gsm8k_id:05d}.json")
    return file_path.exists(), file_path

def process_single_question(model_config, persona_type, persona_id, gsm8k_id, explicit_prompt):
    """Process single GSM8K question with explicit prompt"""
    
    # Check if already exists (skip logic)
    exists, file_path = check_result_exists(model_config, persona_type, persona_id, gsm8k_id)
    if exists:
        return {"status": "skipped", "file_path": str(file_path)}
    
    # Load GSM8K question
    gsm8k_data = load_gsm8k_question(gsm8k_id)
    if not gsm8k_data:
        return {"status": "error", "error": f"Question {gsm8k_id} not found"}
    
    # Build user message: explicit prompt + GSM8K question
    user_message = f"{explicit_prompt['prompt']} {gsm8k_data['question']}\n\nAnswer:"
    
    # Call API with retry
    api_result = call_openrouter_api(model_config, user_message)
    
    # Create result JSON
    result_json = create_result_json(model_config, explicit_prompt, gsm8k_data, user_message, 
                                     api_result, persona_type, persona_id)
    
    # Save result
    file_path = save_result(model_config, persona_type, persona_id, gsm8k_id, result_json)
    
    return {
        "status": "success" if api_result['success'] else "failed",
        "file_path": file_path,
        "retry_count": api_result.get('retry_count', 0),
        "latency": api_result['latency'],
        "error": api_result.get('error')
    }

def main():
    parser = argparse.ArgumentParser(description='Batch Explicit GSM8K Runner with Retry')
    parser.add_argument('--persona_id', type=int, required=True, help='Persona ID (1-3)')
    parser.add_argument('--persona_type', type=str, choices=['woman', 'man'], required=True, help='Persona type: woman or man')
    parser.add_argument('--model_order', type=str, required=True, help='Model order: single (31), range (31-35), or list (31,32,35)')
    parser.add_argument('--questions', type=str, required=True, help='Question range: single (1), range (1-10), or list (1,2,5)')
    
    args = parser.parse_args()
    
    # Parse ranges
    model_orders = parse_range(args.model_order)
    question_ids = parse_range(args.questions)
    
    print("="*80)
    print(f"BATCH EXPLICIT GSM8K RUNNER")
    print("="*80)
    print(f"Persona: {args.persona_type}_explicit_{args.persona_id}")
    print(f"Models: {model_orders}")
    print(f"Questions: {len(question_ids)} questions")
    print("="*80)
    print()
    
    # Load explicit prompt
    explicit_prompt = load_explicit_prompt(args.persona_type, args.persona_id)
    if not explicit_prompt:
        print(f"[ERROR] Explicit prompt {args.persona_id} not found for {args.persona_type}")
        return
    
    print(f"Explicit Prompt: {explicit_prompt['prompt']}")
    print()
    
    # Initialize summary
    summary = {
        "start_time": datetime.now().isoformat(),
        "persona_type": args.persona_type,
        "persona_id": args.persona_id,
        "models": [],
        "total_tasks": 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0
    }
    
    total_tasks = len(model_orders) * len(question_ids)
    current_task = 0
    
    # Process each model
    for model_order in model_orders:
        model = load_model_config(model_order)
        if not model:
            print(f"[ERROR] Model order {model_order} not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{model['order']}] {model['name']}")
        print(f"{'='*80}")
        
        model_summary = {
            "order": model['order'],
            "name": model['name'],
            "model_id": model['model_id'],
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_retries": 0
        }
        
        # Process each question
        for q_id in question_ids:
            current_task += 1
            
            # Show progress
            progress_pct = (current_task / total_tasks) * 100
            print(f"\n[{current_task}/{total_tasks}] ({progress_pct:.1f}%) Question {q_id}...", end=" ")
            
            result = process_single_question(model, args.persona_type, args.persona_id, q_id, explicit_prompt)
            
            if result['status'] == 'skipped':
                print(f"[SKIP]")
                model_summary['skipped'] += 1
                summary['skipped_count'] += 1
            elif result['status'] == 'success':
                retry_info = f" ({result['retry_count']} retries)" if result['retry_count'] > 0 else ""
                print(f"[OK]{retry_info}")
                model_summary['success'] += 1
                model_summary['total_retries'] += result['retry_count']
                summary['success_count'] += 1
            else:
                error_preview = result.get('error', 'Unknown')[:50].encode('ascii', errors='ignore').decode('ascii')
                print(f"[FAIL] {error_preview}")
                model_summary['failed'] += 1
                summary['failed_count'] += 1
            
            # Rate limiting protection (only for non-skipped)
            if result['status'] != 'skipped':
                time.sleep(0.5)
        
        summary['models'].append(model_summary)
        summary['total_tasks'] = current_task
        
        # Model summary
        print(f"\nModel Summary: {model_summary['success']} success, {model_summary['skipped']} skipped, {model_summary['failed']} failed")
    
    # Final summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total: {summary['total_tasks']} tasks")
    print(f"[OK] Success: {summary['success_count']}")
    print(f"[SKIP] Skipped: {summary['skipped_count']}")
    print(f"[FAIL] Failed: {summary['failed_count']}")
    
    # Save summary
    summary['end_time'] = datetime.now().isoformat()
    summary_file = f"logs_batch_{args.persona_type}_explicit_{args.persona_id}_models{model_orders[0]}-{model_orders[-1]}_q{question_ids[0]}-{question_ids[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary log: {summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()
