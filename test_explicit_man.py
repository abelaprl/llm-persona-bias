"""
Test GSM8K dengan Explicit Man Prompt
Format: "I am a man. Solve the following question: [GSM8K question]"

Usage: python test_explicit_man.py
"""
import requests
import json
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

def load_explicit_prompt(prompt_id):
    """Load explicit man prompt"""
    with open('inputs/explicits_man_prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    for p in prompts:
        if p['id'] == prompt_id:
            return p
    return None

def call_openrouter_api(model_config, user_message):
    """Call OpenRouter API"""
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
    
    print(f"\n{'='*80}")
    print(f"REQUEST TO {model_config['model_id']}")
    print(f"{'='*80}")
    print(f"\nMessages:")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content'][:100]}...")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            api_response = response.json()
            return {
                "success": True,
                "response": api_response,
                "latency": latency,
                "error": None
            }
        else:
            return {
                "success": False,
                "response": None,
                "latency": latency,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "response": None,
            "latency": latency,
            "error": str(e)
        }

def main():
    print("="*80)
    print("TEST: GSM8K dengan Explicit Man Prompt #1")
    print("="*80)
    print()
    
    # Configuration
    model_order = 31
    question_id = 1
    prompt_id = 1
    
    # Load data
    model = load_model_config(model_order)
    if not model:
        print(f"[ERROR] Model order {model_order} not found")
        return
    
    gsm8k = load_gsm8k_question(question_id)
    if not gsm8k:
        print(f"[ERROR] Question {question_id} not found")
        return
    
    explicit_prompt = load_explicit_prompt(prompt_id)
    if not explicit_prompt:
        print(f"[ERROR] Explicit prompt {prompt_id} not found")
        return
    
    print(f"Model: [{model['order']}] {model['name']}")
    print(f"Question ID: {question_id}")
    print(f"Explicit Prompt: {explicit_prompt['prompt']}")
    print(f"GSM8K Question: {gsm8k['question'][:100]}...")
    print()
    
    # Build user message: explicit prompt + GSM8K question
    # Format: "I am a man. Solve the following question: [question]\n\nAnswer:"
    user_message = f"{explicit_prompt['prompt']} {gsm8k['question']}\n\nAnswer:"
    
    # Call API
    result = call_openrouter_api(model, user_message)
    
    # Print result
    print(f"\n{'='*80}")
    print(f"RESULT")
    print(f"{'='*80}")
    
    if result['success']:
        content = result['response']['choices'][0]['message'].get('content', '')
        usage = result['response'].get('usage', {})
        
        print(f"\n[OK] SUCCESS")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Tokens: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}, total={usage.get('total_tokens')}")
        print(f"\nResponse:")
        print(content)
        
        # Save result
        model_name = model['model_id'].replace('/', '_')
        output_dir = Path(f"result/{model_name}/man_explicit_{prompt_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "metadata": {
                "run_id": f"{model_name}_man_explicit_{prompt_id}_gsm8k_{question_id:05d}",
                "experiment_type": "gsm8k_bias_study",
                "scenario": "A",
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "provider": "openrouter",
                    "model_id": model['model_id'],
                    "model_name": model['name'],
                    "api_endpoint": "https://openrouter.ai/api/v1/chat/completions"
                }
            },
            "input": {
                "question_id": question_id,
                "dataset": "gsm8k_test",
                "question": gsm8k['question'],
                "ground_truth": gsm8k.get('answer', ''),
                "persona": {
                    "id": prompt_id,
                    "type": "man_explicit",
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
                    "model": model['model_id']
                }
            },
            "response": {
                "success": True,
                "content": content,
                "raw_api_response": result['response'],
                "latency": result['latency']
            }
        }
        
        output_file = output_dir / f"gsm8k_{question_id:05d}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved to: {output_file}")
    else:
        print(f"\n[FAIL] ERROR: {result['error']}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
