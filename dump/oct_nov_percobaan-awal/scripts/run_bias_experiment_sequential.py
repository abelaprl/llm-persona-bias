#!/usr/bin/env python3
"""
Run bias experiment SEQUENTIALLY across all 12 models.
Each model runs one after another to avoid rate limits.
Results saved to separate files: bias_experiment_results_model_XX_<model_name>.json
"""
import json
import re
import time
import random
import requests
from pathlib import Path
from datetime import datetime

# Fallback mapping for models that return 404 (no endpoints)
FALLBACK_MODELS = {
    'openrouter/polaris-alpha': 'meta-llama/llama-3.3-8b-instruct:free',
}

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
        answer = re.sub(r'[<>$]', '', answer)
        return answer
    return text.strip()

def call_api(messages, model_id, api_key, retry_count=3):
    """Call chat API and return response with retry logic."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/abelaprl/TugasAkhir",
        "X-Title": "TugasAkhir Bias Experiment"
    }
    payload = {
        "model": model_id,
        "messages": messages
    }
    current_model = model_id
    
    for attempt in range(retry_count):
        start = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            latency = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return {
                    'status': 200,
                    'content': content,
                    'latency_sec': latency,
                    'used_model': current_model
                }
            elif response.status_code == 404:
                # Try fallback model if defined
                fb = FALLBACK_MODELS.get(payload["model"])
                if fb:
                    print(f"      🔁 404 for {payload['model']} → trying fallback {fb}...")
                    payload["model"] = fb
                    current_model = fb
                    continue
            elif response.status_code == 429 and attempt < retry_count - 1:
                # Rate limit - wait longer and retry
                wait_time = 10 * (attempt + 1)
                print(f"      ⏳ Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    'status': response.status_code,
                    'error': response.text,
                    'latency_sec': latency,
                    'used_model': current_model
                }
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 5 * (attempt + 1)
                print(f"      ⚠️ Error: {str(e)[:50]}..., retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return {
                'status': 0,
                'error': str(e),
                'latency_sec': time.time() - start,
                'used_model': current_model
            }
    
    return {
        'status': 0,
        'error': 'Max retries exceeded',
        'latency_sec': 0,
        'used_model': current_model
    }

def run_experiment_for_model(model_entry, base_path):
    """Run experiment for a single model."""
    model_order = model_entry['order']
    model_id = model_entry['model_id']
    api_key = model_entry['api_key_alias']
    
    print("\n" + "=" * 70)
    print(f"🚀 MODEL {model_order:02d}/{12}: {model_id}")
    print("=" * 70)
    
    # Load warmups
    implicit_warmups, explicit_warmups = load_warmup_prompts(base_path)
    
    # Load questions
    gsm_questions = load_gold_data(base_path / 'data' / 'gsm8k_gold.jsonl', limit=20)
    mmlu_questions = load_gold_data(base_path / 'data' / 'mmlu_gold.jsonl', limit=20)
    
    results = []
    total_calls = 0
    start_time = time.time()
    
    # Process GSM8K
    print(f"\n📊 GSM8K Dataset (20 questions × 3 conditions)")
    print("-" * 70)
    
    for q_idx, q_data in enumerate(gsm_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        gold_answer = re.search(r'####\s*(.+)', q_data['answer'])
        gold_answer = gold_answer.group(1).strip() if gold_answer else None
        
        print(f"\n[{q_idx}/20] {question_id}")
        
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
                        print(f"  🔄 Warmup {w_idx}/5 ✓ [{warmup_response['latency_sec']:.1f}s]", end='')
                    else:
                        print(f"  ⚠️ Warmup {w_idx}/5 failed: {warmup_response.get('status', 'unknown')}")
                        break
                    
                    time.sleep(2.0)  # Delay between warmups
                
                if warmup_responses:
                    print()  # Newline after warmups
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
                        print(f"  🔄 Warmup {w_idx}/5 ✓ [{warmup_response['latency_sec']:.1f}s]", end='')
                    else:
                        print(f"  ⚠️ Warmup {w_idx}/5 failed: {warmup_response.get('status', 'unknown')}")
                        break
                    
                    time.sleep(2.0)
                
                if warmup_responses:
                    print()
                warmup_count = len(warmup_responses)
            
            messages.append({"role": "user", "content": final_prompt})
            response = call_api(messages, model_id, api_key)
            total_calls += 1
            
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
                'model_order': model_order,
                'model_id': model_id,
                'resolved_model_id': response.get('used_model', model_id),
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
            
            time.sleep(2.5)  # INCREASED delay between conditions
    
    # Process MMLU
    print(f"\n📚 MMLU Dataset (20 questions × 3 conditions)")
    print("-" * 70)
    
    for q_idx, q_data in enumerate(mmlu_questions, 1):
        question_id = q_data['id']
        question = q_data['question']
        choices = {'A': q_data['A'], 'B': q_data['B'], 'C': q_data['C'], 'D': q_data['D']}
        gold_answer_idx = q_data['answer']
        gold_answer = ['A', 'B', 'C', 'D'][gold_answer_idx]
        
        print(f"\n[{q_idx}/20] {question_id}")
        
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
                        print(f"  🔄 Warmup {w_idx}/5 ✓ [{warmup_response['latency_sec']:.1f}s]", end='')
                    else:
                        print(f"  ⚠️ Warmup {w_idx}/5 failed: {warmup_response.get('status', 'unknown')}")
                        break
                    
                    time.sleep(2.0)
                
                if warmup_responses:
                    print()
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
                        print(f"  🔄 Warmup {w_idx}/5 ✓ [{warmup_response['latency_sec']:.1f}s]", end='')
                    else:
                        print(f"  ⚠️ Warmup {w_idx}/5 failed: {warmup_response.get('status', 'unknown')}")
                        break
                    
                    time.sleep(2.0)
                
                if warmup_responses:
                    print()
                warmup_count = len(warmup_responses)
            
            messages.append({"role": "user", "content": final_prompt})
            response = call_api(messages, model_id, api_key)
            total_calls += 1
            
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
                'model_order': model_order,
                'model_id': model_id,
                'resolved_model_id': response.get('used_model', model_id),
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
            
            time.sleep(2.5)  # INCREASED delay between conditions
    
    # Save results for this model
    model_name_clean = model_id.replace('/', '_').replace(':', '_')
    output_file = base_path / 'results' / f'bias_experiment_results_model_{model_order:02d}_{model_name_clean}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"✅ Model {model_order:02d} Complete!")
    print(f"📊 {total_calls} API calls in {elapsed/60:.1f} minutes")
    print(f"📁 Saved to: {output_file.name}")
    print("=" * 70)
    
    # Quick analysis
    print(f"\n📈 Accuracy Summary:")
    
    for dataset in ['gsm8k', 'mmlu']:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        print(f"\n  {dataset.upper()}:")
        
        for condition in ['control', 'implicit', 'explicit']:
            cond_results = [r for r in dataset_results if r['condition'] == condition]
            correct = sum(1 for r in cond_results if r['is_correct'])
            total = len(cond_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"    {condition:8s}: {correct}/{total} = {accuracy:.1f}%")
    
    return results

def main():
    import sys
    
    base = Path(__file__).parent.parent
    
    # Load all models
    with open(base / 'config' / 'model_keys.json', 'r') as f:
        models = json.load(f)
    
    # Get starting model from command line argument (default = 2, skip model 1)
    start_model = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    # Filter models: only run the specified model
    target_models = [m for m in models if m['order'] == start_model]
    
    if not target_models:
        print(f"❌ Model {start_model} not found in config!")
        return
    
    print("=" * 70)
    print(f"🚀 SEQUENTIAL BIAS EXPERIMENT - MODEL {start_model} ONLY")
    print("=" * 70)
    print(f"📋 Running 1 model:")
    print(f"   Model {target_models[0]['order']:02d}: {target_models[0]['model_id']}")
    print(f"\n📊 Each model: 520 API calls (20 questions × 13 calls per question)")
    print(f"⏱️  Estimated time: ~15-20 minutes")
    print(f"🔄 Retry logic enabled, delays to avoid rate limits\n")
    
    overall_start = time.time()
    
    # Run the single model
    model_entry = target_models[0]
    try:
        run_experiment_for_model(model_entry, base)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user at Model {model_entry['order']:02d}")
        print(f"📁 Partial results saved.")
    except Exception as e:
        print(f"\n\n❌ Error in Model {model_entry['order']:02d}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print(f"✅ MODEL {start_model} COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {overall_elapsed/60:.1f} minutes")
    print(f"📁 Results saved to: results/bias_experiment_results_model_{start_model:02d}_<model_name>.json")
    print(f"\n� To run next model: python3 scripts/run_bias_experiment_sequential.py {start_model + 1}")
    print("=" * 70)

if __name__ == '__main__':
    main()
