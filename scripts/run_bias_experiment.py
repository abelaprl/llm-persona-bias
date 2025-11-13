#!/usr/bin/env python3
"""
Run bias experiment: test if implicit/explicit feminine bias affects accuracy.
Uses ITERATIVE warmup: 5 real LLM responses to warmup questions before final question.
20 questions × (1 control + 6 implicit + 6 explicit) × 2 datasets = 520 API calls.
"""
import json
import re
import time
import random
import requests
from pathlib import Path
from datetime import datetime

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

def load_warmup_prompts(base_path):
    """Load warmup prompts."""
    implicit_path = base_path / 'data' / 'question_women_en_implicit.txt'
    explicit_path = base_path / 'data' / 'question_women_en_explicit.txt'
    
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

def call_api(messages, model_id, api_key):
    """Call chat API and return response."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": messages
    }
    
    start = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            return {
                'status': 200,
                'content': content,
                'latency_sec': latency,
                'raw_json': data
            }
        else:
            return {
                'status': response.status_code,
                'error': response.text,
                'latency_sec': latency
            }
    except Exception as e:
        return {
            'status': 0,
            'error': str(e),
            'latency_sec': time.time() - start
        }

def run_experiment():
    base = Path(__file__).parent.parent
    
    # Load config
    with open(base / 'config' / 'model_keys.json', 'r') as f:
        models = json.load(f)
    
    # Use first model
    model_entry = models[0]
    model_id = model_entry['model_id']
    api_key = model_entry['api_key_alias']
    
    print(f"🚀 Running bias experiment with model: {model_id}")
    print(f"📊 ITERATIVE warmup: 5 real LLM responses per biased condition")
    print(f"📊 Target: 20 questions × (1 control + 6 implicit + 6 explicit) × 2 datasets = 520 API calls\n")
    
    # Load warmups
    implicit_warmups, explicit_warmups = load_warmup_prompts(base)
    print(f"✅ Loaded {len(implicit_warmups)} implicit warmups")
    print(f"✅ Loaded {len(explicit_warmups)} explicit warmups\n")
    
    # Load questions
    gsm_questions = load_gold_data(base / 'data' / 'gsm8k_gold.jsonl', limit=20)
    mmlu_questions = load_gold_data(base / 'data' / 'mmlu_gold.jsonl', limit=20)
    
    print(f"✅ Loaded {len(gsm_questions)} GSM8K questions")
    print(f"✅ Loaded {len(mmlu_questions)} MMLU questions\n")
    
    results = []
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
                    warmup_response = call_api(messages, model_id, api_key)
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
                    warmup_response = call_api(messages, model_id, api_key)
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
            response = call_api(messages, model_id, api_key)
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
                print(f"  {condition:8s}: ⚠️  API Error {response['status']}")
            
            # Save result
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
                    warmup_response = call_api(messages, model_id, api_key)
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
                    warmup_response = call_api(messages, model_id, api_key)
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
            response = call_api(messages, model_id, api_key)
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
                print(f"  {condition:8s}: ⚠️  API Error {response['status']}")
            
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
    output_file = base / 'results' / 'bias_experiment_results.json'
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
    run_experiment()
