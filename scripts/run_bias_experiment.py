#!/usr/bin/env python3
"""
Run bias experiment: test if implicit/explicit feminine bias affects accuracy.
Uses ITERATIVE warmup: 5 real LLM responses to warmup questions before final question.
20 questions × (1 control + 6 implicit + 6 explicit) × 2 datasets = 520 API calls.
"""
import argparse
import json
import re
import time
import random
import requests
import os
from pathlib import Path
from datetime import datetime

# Try to import Groq SDK if available
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    Groq = None
    HAS_GROQ = False

# Retry/backoff defaults (can be overridden via CLI)
MAX_RETRIES = 3
BACKOFF_BASE = 0.5  # seconds
BACKOFF_FACTOR = 2.0
JITTER = 0.1  # seconds

def load_gold_data(gold_path, limit=20):
    """Load first N questions from gold JSONL."""
    questions = []
    with open(gold_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if line.strip():
                questions.append(json.loads(line))
    return questions

def load_warmup_prompts(base_path, persona='women'):
    """Load warmup prompts for a persona (women or men)."""
    implicit_path = base_path / 'data' / f'question_{persona}_en_implicit.txt'
    explicit_path = base_path / 'data' / f'question_{persona}_en_explicit.txt'

    with open(implicit_path, 'r', encoding='utf-8') as f:
        implicit = [line.strip() for line in f if line.strip()]

    with open(explicit_path, 'r', encoding='utf-8') as f:
        explicit = [line.strip() for line in f if line.strip()]

    return implicit, explicit

def build_gsm8k_prompt(question):
    """Build GSM8K prompt."""
    return f"""Solve this math problem step by step and provide your final numerical answer.

Question: {question}

Respond with your final answer in the format:
Final Answer: <number>"""

def build_mmlu_prompt(question, choices):
    """Build MMLU prompt."""
    return f"""Choose the correct answer and respond with only:
Final Answer: <A/B/C/D>

Question:
{question}
A. {choices['A']}
B. {choices['B']}
C. {choices['C']}
D. {choices['D']}"""

def extract_final_answer(text):
    """Extract answer after 'Final Answer:' marker."""
    match = re.search(r'Final Answer:\s*(.+)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up common artifacts
        answer = re.sub(r'[<>]', '', answer)
        return answer
    return text.strip()

def call_api(messages, model_entry):
    """Call provider-specific chat API and return response.

    model_entry: dict containing at least 'model_id' and 'api_key_alias'.
    Supports provider detection for 'groq' (model_id startswith 'groq/') and
    falls back to OpenRouter-compatible endpoint.
    """
    model_id = model_entry.get('model_id')
    api_key = model_entry.get('api_key_alias')
    attempt = 1
    while attempt <= MAX_RETRIES:
        start = time.time()

        # Provider: Groq (model_id like 'groq/xxxx')
        if isinstance(model_id, str) and model_id.startswith('groq/'):
            model_name = model_id.split('/', 1)[1]
            # Prefer environment variable for key; fall back to config value
            groq_key = os.getenv('GROQ_API_KEY') or api_key

            # If SDK is available and we have a key, use it (cleanest integration)
            if HAS_GROQ and groq_key:
                try:
                    client = Groq(api_key=groq_key)
                    resp = client.chat.completions.create(model=model_name, messages=messages)
                    latency = time.time() - start
                    # Try common access patterns
                    content = None
                    try:
                        content = resp.choices[0].message.content
                    except Exception:
                        try:
                            content = resp['choices'][0]['message']['content']
                        except Exception:
                            content = str(resp)
                    return {'status': 200, 'content': content, 'latency_sec': latency, 'raw_json': resp}
                except Exception as e:
                    result = {'status': 0, 'error': str(e), 'latency_sec': time.time() - start}
                    # transient error => consider retry
                    transient = True
            else:
                transient = False
        else:
            # Not a Groq model; don't treat as Groq provider here (will fallthrough to fallback below)
            transient = False

        # SDK not available or key missing — fall back to HTTP path (endpoint can be overridden in config)
        endpoint = model_entry.get('endpoint') or 'https://api.groq.com/openai/v1'
        url = f"{endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": model_name, "messages": messages}

        print(f"⚠️  Using Groq HTTP endpoint {url} — if this fails, set a correct 'endpoint' in model entry or install the 'groq' SDK and set GROQ_API_KEY env var.")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            latency = time.time() - start
            if response.status_code == 200:
                data = response.json()
                # Try to extract assistant content robustly
                content = None
                if isinstance(data, dict):
                    if 'choices' in data and data['choices']:
                        choice = data['choices'][0]
                        if isinstance(choice, dict) and 'message' in choice and isinstance(choice['message'], dict):
                            content = choice['message'].get('content')
                        elif 'text' in choice:
                            content = choice.get('text')
                    elif 'output' in data:
                        out = data['output']
                        if isinstance(out, list) and out:
                            content = out[0].get('content') if isinstance(out[0], dict) else str(out[0])
                content = content or json.dumps(data)
                return {'status': 200, 'content': content, 'latency_sec': latency, 'raw_json': data}
            else:
                result = {'status': response.status_code, 'error': response.text, 'latency_sec': latency}
                transient = response.status_code == 429
        except Exception as e:
            result = {'status': 0, 'error': str(e), 'latency_sec': time.time() - start}
            transient = True

        # If we get here, the attempt didn't return success. Decide whether to retry.
        if attempt < MAX_RETRIES and transient:
            sleep_for = BACKOFF_BASE * (BACKOFF_FACTOR ** (attempt - 1)) + random.random() * JITTER
            print(f"    ⏳ Transient error (attempt {attempt}/{MAX_RETRIES}), sleeping {sleep_for:.2f}s before retry: {result.get('error') or result.get('status')}")
            time.sleep(sleep_for)
            attempt += 1
            continue
        else:
            return result

    # Fallback: OpenRouter-compatible endpoint (existing behavior)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages}
    # Wrap fallback call with retry logic too
    attempt = 1
    while attempt <= MAX_RETRIES:
        start = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            latency = time.time() - start
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return {'status': 200, 'content': content, 'latency_sec': latency, 'raw_json': data}
            else:
                result = {'status': response.status_code, 'error': response.text, 'latency_sec': latency}
                transient = response.status_code == 429
        except Exception as e:
            result = {'status': 0, 'error': str(e), 'latency_sec': time.time() - start}
            transient = True

        if attempt < MAX_RETRIES and transient:
            sleep_for = BACKOFF_BASE * (BACKOFF_FACTOR ** (attempt - 1)) + random.random() * JITTER
            print(f"    ⏳ Transient error (fallback attempt {attempt}/{MAX_RETRIES}), sleeping {sleep_for:.2f}s before retry: {result.get('error') or result.get('status')}")
            time.sleep(sleep_for)
            attempt += 1
            continue
        else:
            return result

def run_experiment(persona='women', limit=20, model_index=None, model_id_override=None):
    base = Path(__file__).parent.parent
    
    # Load config
    with open(base / 'config' / 'model_keys.json', 'r') as f:
        models = json.load(f)
    
    # Select model according to parameters (default: first model)
    selected_idx = None
    model_entry = None
    if model_index is not None:
        if model_index < 0 or model_index >= len(models):
            print(f"❌ model_index {model_index} out of range (0..{len(models)-1})")
            return []
        model_entry = models[model_index]
        selected_idx = model_index
    elif model_id_override:
        for i, m in enumerate(models):
            if m.get('model_id') == model_id_override or m.get('name') == model_id_override:
                model_entry = m
                selected_idx = i
                break
        if model_entry is None:
            print(f"❌ model_id '{model_id_override}' not found in config/model_keys.json")
            return []
    else:
        model_entry = models[0]
        selected_idx = 0

    model_id = model_entry['model_id']
    api_key = model_entry['api_key_alias']
    
    name = model_entry.get('name', '')
    print(f"🚀 Running bias experiment with model: {model_id} (name: {name}, index: {selected_idx}) (persona: {persona})")
    print(f"📊 ITERATIVE warmup: 5 real LLM responses per biased condition")
    print(f"📊 Target: 20 questions × (1 control + 6 implicit + 6 explicit) × 2 datasets = 520 API calls\n")
    
    # Load warmups for persona
    implicit_warmups, explicit_warmups = load_warmup_prompts(base, persona)
    print(f"✅ Loaded {len(implicit_warmups)} implicit warmups")
    print(f"✅ Loaded {len(explicit_warmups)} explicit warmups\n")
    
    # Load questions
    gsm_questions = load_gold_data(base / 'data' / 'gsm8k_gold.jsonl', limit=limit)
    mmlu_questions = load_gold_data(base / 'data' / 'mmlu_gold.jsonl', limit=limit)
    
    print(f"✅ Loaded {len(gsm_questions)} GSM8K questions")
    print(f"✅ Loaded {len(mmlu_questions)} MMLU questions\n")
    
    # Prepare results and support resuming from a previous run
    output_file = base / 'results' / f'bias_experiment_results_{persona}.json'
    existing_results = []
    completed = set()
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            for r in existing_results:
                completed.add((r.get('dataset'), r.get('question_id'), r.get('condition')))
            print(f"⚠️ Detected existing results file '{output_file.name}' with {len(existing_results)} rows — will resume and skip completed items")
        except Exception as e:
            print(f"⚠️ Could not read existing results file: {e}")

    results = existing_results.copy()
    total_calls = 0
    
    # Process GSM8K
    print("=" * 60)
    print("🧮 Processing GSM8K questions...")
    print("=" * 60)
    
    for q_idx, q_data in enumerate(gsm_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        gold_answer = re.search(r'####\s*(.+)', q_data['answer'])
        gold_answer = gold_answer.group(1).strip() if gold_answer else None
        
        print(f"\n[{q_idx}/20] Question: {question_id}")
        
        # Build final prompt
        final_prompt = build_gsm8k_prompt(question)
        
        # Test 3 conditions
        for condition in ['control', 'implicit', 'explicit']:
            # Skip if this exact dataset/question/condition already has a saved result
            if ( 'gsm8k', question_id, condition) in completed:
                print(f"  {condition:8s}: ⏭️ Skipping — result already present in {output_file.name}")
                continue
            messages = []
            warmup_count = 0
            warmup_prompts_used = []
            warmup_responses = []
            
            if condition == 'implicit':
                # ITERATIVE warmup: 5 random implicit questions with REAL LLM responses
                selected_warmups = random.sample(implicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for warmup_q in selected_warmups:
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_entry)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"    🔄 Warmup {len(warmup_responses)}/5 complete [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"    ⚠️  Warmup {len(warmup_responses)+1}/5 failed")
                        break
                    
                    time.sleep(0.3)  # Rate limiting between warmups
                
                warmup_count = len(warmup_responses)
                
            elif condition == 'explicit':
                # ITERATIVE warmup: 5 random explicit questions with REAL LLM responses
                selected_warmups = random.sample(explicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for warmup_q in selected_warmups:
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_entry)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"    🔄 Warmup {len(warmup_responses)}/5 complete [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"    ⚠️  Warmup {len(warmup_responses)+1}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
            
            # Add final question
            messages.append({"role": "user", "content": final_prompt})
            
            # Call API for final answer
            response = call_api(messages, model_entry)
            total_calls += 1
            
            # Extract answer
            if response['status'] == 200:
                model_answer = extract_final_answer(response['content'])
                is_correct = (str(model_answer).strip() == str(gold_answer).strip())
                status_icon = "✅" if is_correct else "❌"
                print(f"  {condition:8s}: {status_icon} {model_answer} (gold: {gold_answer}) [{response['latency_sec']:.1f}s]")
            else:
                model_answer = None
                is_correct = False
                err = response.get('error')
                print(f"  {condition:8s}: ⚠️  API Error {response['status']} {err if err else ''}")
            
            # Save result (avoid duplicating if resumed)
            key = ('gsm8k', question_id, condition)
            if key not in completed:
                results.append({
                'dataset': 'gsm8k',
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'condition': condition,
                'warmup_count': warmup_count,
                'warmup_prompts': warmup_prompts_used,
                'warmup_responses': warmup_responses,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'latency_sec': response.get('latency_sec', 0),
                'status': response['status'],
                'timestamp': datetime.now().isoformat()
                })
                completed.add(key)
            
            # Rate limiting
            time.sleep(0.5)
    
    # Process MMLU
    print("\n" + "=" * 60)
    print("📚 Processing MMLU questions...")
    print("=" * 60)
    
    for q_idx, q_data in enumerate(mmlu_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        choices = {'A': q_data['A'], 'B': q_data['B'], 'C': q_data['C'], 'D': q_data['D']}
        gold_answer_idx = q_data['answer']
        gold_answer = ['A', 'B', 'C', 'D'][gold_answer_idx]
        
        print(f"\n[{q_idx}/20] Question: {question_id}")
        
        # Build final prompt
        final_prompt = build_mmlu_prompt(question, choices)
        
        # Test 3 conditions
        for condition in ['control', 'implicit', 'explicit']:
            messages = []
            warmup_count = 0
            warmup_prompts_used = []
            warmup_responses = []
            
            if condition == 'implicit':
                # ITERATIVE warmup: 5 random implicit questions with REAL LLM responses
                selected_warmups = random.sample(implicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for warmup_q in selected_warmups:
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_entry)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"    🔄 Warmup {len(warmup_responses)}/5 complete [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"    ⚠️  Warmup {len(warmup_responses)+1}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
                
            elif condition == 'explicit':
                # ITERATIVE warmup: 5 random explicit questions with REAL LLM responses
                selected_warmups = random.sample(explicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for warmup_q in selected_warmups:
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_entry)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"    🔄 Warmup {len(warmup_responses)}/5 complete [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"    ⚠️  Warmup {len(warmup_responses)+1}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
            
            messages.append({"role": "user", "content": final_prompt})
            
            # Call API for final answer
            response = call_api(messages, model_entry)
            total_calls += 1
            
            # Extract answer
            if response['status'] == 200:
                model_answer = extract_final_answer(response['content'])
                is_correct = (str(model_answer).strip() == str(gold_answer).strip())
                status_icon = "✅" if is_correct else "❌"
                print(f"  {condition:8s}: {status_icon} {model_answer} (gold: {gold_answer}) [{response['latency_sec']:.1f}s]")
            else:
                model_answer = None
                is_correct = False
                err = response.get('error')
                print(f"  {condition:8s}: ⚠️  API Error {response['status']} {err if err else ''}")
            
            results.append({
                'dataset': 'mmlu',
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'condition': condition,
                'warmup_count': warmup_count,
                'warmup_prompts': warmup_prompts_used,
                'warmup_responses': warmup_responses,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'latency_sec': response.get('latency_sec', 0),
                'status': response['status'],
                'timestamp': datetime.now().isoformat()
            })
            
            time.sleep(0.5)
    
    # Save results
    output_file = base / 'results' / f'bias_experiment_results_{persona}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Experiment complete! {total_calls} API calls made")
    print(f"📁 Results saved to: {output_file}")
    print("=" * 60)
    
    # Quick analysis
    print("\n📊 Quick Analysis:")
    
    for dataset in ['gsm8k', 'mmlu']:
        print(f"\n{dataset.upper()}:")
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        for condition in ['control', 'implicit', 'explicit']:
            cond_results = [r for r in dataset_results if r['condition'] == condition]
            correct = sum(1 for r in cond_results if r['is_correct'])
            total = len(cond_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"  {condition:8s}: {correct}/{total} = {accuracy:.1f}%")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run bias experiment for a persona (women|men)')
    parser.add_argument('--persona', choices=['women', 'men'], default='women', help='Persona to use for warmup prompts')
    parser.add_argument('--limit', type=int, default=20, help='Number of gold questions per dataset to run (default 20)')
    parser.add_argument('--model-index', type=int, default=None, help='Index of model to use from config/model_keys.json (0-based)')
    parser.add_argument('--model-id', type=str, default=None, help='Model id or name to select from config/model_keys.json')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help='Max retry attempts for transient errors (default 3)')
    parser.add_argument('--backoff-base', type=float, default=BACKOFF_BASE, help='Base backoff seconds (default 0.5)')
    parser.add_argument('--backoff-factor', type=float, default=BACKOFF_FACTOR, help='Backoff multiplier per attempt (default 2.0)')
    args = parser.parse_args()

    # Set runtime retry/backoff parameters from CLI
    MAX_RETRIES = args.max_retries
    BACKOFF_BASE = args.backoff_base
    BACKOFF_FACTOR = args.backoff_factor

    run_experiment(persona=args.persona, limit=args.limit, model_index=args.model_index, model_id_override=args.model_id)
