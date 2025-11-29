#!/usr/bin/env python3
"""
Run bias experiment in PARALLEL across all 12 models.
Each model runs independently with iterative warmup.
Results saved to separate files: bias_experiment_results_model_XX.json
"""
import json
import re
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from multiprocessing import Process
import sys

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

def run_experiment_for_model(model_entry, model_index, base_path):
    """Run experiment for a single model."""
    model_order = model_entry['order']
    model_id = model_entry['model_id']
    api_key = model_entry['api_key_alias']
    
    print(f"[Model {model_order:02d}] 🚀 Starting experiment: {model_id}")
    print(f"[Model {model_order:02d}] 📊 ITERATIVE warmup: 5 real LLM responses per biased condition")
    print(f"[Model {model_order:02d}] 📊 Target: 520 API calls\n")
    
    # Load warmups
    implicit_warmups, explicit_warmups = load_warmup_prompts(base_path)
    
    # Load questions
    gsm_questions = load_gold_data(base_path / 'data' / 'gsm8k_gold.jsonl', limit=20)
    mmlu_questions = load_gold_data(base_path / 'data' / 'mmlu_gold.jsonl', limit=20)
    
    results = []
    total_calls = 0
    
    # Process GSM8K
    print(f"[Model {model_order:02d}] " + "=" * 50)
    print(f"[Model {model_order:02d}] 🧮 Processing GSM8K questions...")
    print(f"[Model {model_order:02d}] " + "=" * 50)
    
    for q_idx, q_data in enumerate(gsm_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        gold_answer = re.search(r'####\s*(.+)', q_data['answer'])
        gold_answer = gold_answer.group(1).strip() if gold_answer else None
        
        print(f"[Model {model_order:02d}] [{q_idx}/20] {question_id}")
        
        final_prompt = build_gsm8k_prompt(question)
        
        for condition in ['control', 'implicit', 'explicit']:
            messages = []
            warmup_count = 0
            warmup_prompts_used = []
            warmup_responses = []
            
            if condition == 'implicit':
                selected_warmups = random.sample(implicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for w_idx, warmup_q in enumerate(selected_warmups, 1):
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_id, api_key)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"[Model {model_order:02d}]   🔄 Warmup {w_idx}/5 [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"[Model {model_order:02d}]   ⚠️  Warmup {w_idx}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
                
            elif condition == 'explicit':
                selected_warmups = random.sample(explicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for w_idx, warmup_q in enumerate(selected_warmups, 1):
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_id, api_key)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"[Model {model_order:02d}]   🔄 Warmup {w_idx}/5 [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"[Model {model_order:02d}]   ⚠️  Warmup {w_idx}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
            
            messages.append({"role": "user", "content": final_prompt})
            response = call_api(messages, model_id, api_key)
            total_calls += 1
            
            if response['status'] == 200:
                model_answer = extract_final_answer(response['content'])
                is_correct = (str(model_answer).strip() == str(gold_answer).strip())
                status_icon = "✅" if is_correct else "❌"
                print(f"[Model {model_order:02d}]   {condition:8s}: {status_icon} {model_answer} (gold: {gold_answer}) [{response['latency_sec']:.1f}s]")
            else:
                model_answer = None
                is_correct = False
                print(f"[Model {model_order:02d}]   {condition:8s}: ⚠️  API Error {response['status']}")
            
            results.append({
                'model_order': model_order,
                'model_id': model_id,
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
            
            time.sleep(0.5)
    
    # Process MMLU
    print(f"\n[Model {model_order:02d}] " + "=" * 50)
    print(f"[Model {model_order:02d}] 📚 Processing MMLU questions...")
    print(f"[Model {model_order:02d}] " + "=" * 50)
    
    for q_idx, q_data in enumerate(mmlu_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        choices = {'A': q_data['A'], 'B': q_data['B'], 'C': q_data['C'], 'D': q_data['D']}
        gold_answer_idx = q_data['answer']
        gold_answer = ['A', 'B', 'C', 'D'][gold_answer_idx]
        
        print(f"[Model {model_order:02d}] [{q_idx}/20] {question_id}")
        
        final_prompt = build_mmlu_prompt(question, choices)
        
        for condition in ['control', 'implicit', 'explicit']:
            messages = []
            warmup_count = 0
            warmup_prompts_used = []
            warmup_responses = []
            
            if condition == 'implicit':
                selected_warmups = random.sample(implicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for w_idx, warmup_q in enumerate(selected_warmups, 1):
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_id, api_key)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"[Model {model_order:02d}]   🔄 Warmup {w_idx}/5 [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"[Model {model_order:02d}]   ⚠️  Warmup {w_idx}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
                
            elif condition == 'explicit':
                selected_warmups = random.sample(explicit_warmups, 5)
                warmup_prompts_used = selected_warmups.copy()
                
                for w_idx, warmup_q in enumerate(selected_warmups, 1):
                    messages.append({"role": "user", "content": warmup_q})
                    warmup_response = call_api(messages, model_id, api_key)
                    total_calls += 1
                    
                    if warmup_response['status'] == 200:
                        warmup_content = warmup_response['content']
                        messages.append({"role": "assistant", "content": warmup_content})
                        warmup_responses.append(warmup_content)
                        print(f"[Model {model_order:02d}]   🔄 Warmup {w_idx}/5 [{warmup_response['latency_sec']:.1f}s]")
                    else:
                        print(f"[Model {model_order:02d}]   ⚠️  Warmup {w_idx}/5 failed")
                        break
                    
                    time.sleep(0.3)
                
                warmup_count = len(warmup_responses)
            
            messages.append({"role": "user", "content": final_prompt})
            response = call_api(messages, model_id, api_key)
            total_calls += 1
            
            if response['status'] == 200:
                model_answer = extract_final_answer(response['content'])
                is_correct = (str(model_answer).strip() == str(gold_answer).strip())
                status_icon = "✅" if is_correct else "❌"
                print(f"[Model {model_order:02d}]   {condition:8s}: {status_icon} {model_answer} (gold: {gold_answer}) [{response['latency_sec']:.1f}s]")
            else:
                model_answer = None
                is_correct = False
                print(f"[Model {model_order:02d}]   {condition:8s}: ⚠️  API Error {response['status']}")
            
            results.append({
                'model_order': model_order,
                'model_id': model_id,
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
    
    # Save results for this model
    # Clean model_id for filename (remove slashes and special chars)
    model_name_clean = model_id.replace('/', '_').replace(':', '_')
    output_file = base_path / 'results' / f'bias_experiment_results_model_{model_order:02d}_{model_name_clean}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Model {model_order:02d}] " + "=" * 50)
    print(f"[Model {model_order:02d}] ✅ Complete! {total_calls} API calls")
    print(f"[Model {model_order:02d}] 📁 Saved to: {output_file.name}")
    print(f"[Model {model_order:02d}] " + "=" * 50)
    
    # Quick analysis
    print(f"\n[Model {model_order:02d}] 📊 Quick Analysis:")
    
    for dataset in ['gsm8k', 'mmlu']:
        print(f"[Model {model_order:02d}] {dataset.upper()}:")
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        for condition in ['control', 'implicit', 'explicit']:
            cond_results = [r for r in dataset_results if r['condition'] == condition]
            correct = sum(1 for r in cond_results if r['is_correct'])
            total = len(cond_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"[Model {model_order:02d}]   {condition:8s}: {correct}/{total} = {accuracy:.1f}%")
    
    return results

def main():
    base = Path(__file__).parent.parent
    
    # Load all models
    with open(base / 'config' / 'model_keys.json', 'r') as f:
        models = json.load(f)
    
    print("=" * 70)
    print("🚀 PARALLEL BIAS EXPERIMENT - 12 MODELS")
    print("=" * 70)
    print(f"📋 Loaded {len(models)} models from config")
    print(f"📊 Each model: 520 API calls (20 questions × 13 calls per question)")
    print(f"💰 Total: {len(models) * 520} API calls across all models")
    print(f"⏱️  Running in parallel - check progress below\n")
    
    # Create process for each model
    processes = []
    for i, model_entry in enumerate(models):
        p = Process(target=run_experiment_for_model, args=(model_entry, i, base))
        processes.append(p)
        p.start()
        print(f"✅ Started process for Model {model_entry['order']:02d}: {model_entry['model_id']}")
        time.sleep(0.5)  # Stagger starts slightly
    
    print("\n" + "=" * 70)
    print("⏳ All processes started. Waiting for completion...")
    print("=" * 70 + "\n")
    
    # Wait for all to complete
    for i, p in enumerate(processes):
        p.join()
        print(f"\n✅ Model {models[i]['order']:02d} process completed!")
    
    print("\n" + "=" * 70)
    print("🎉 ALL MODELS COMPLETE!")
    print("=" * 70)
    print(f"📁 Results saved to: results/bias_experiment_results_model_XX_<model_name>.json")
    print(f"📁 Total files: 12 (one per model)")
    print("=" * 70)

if __name__ == '__main__':
    main()
