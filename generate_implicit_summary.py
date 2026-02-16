"""
Generate CSV Summary untuk Implicit Results
Output:
1. CSV per model per persona dengan akurasi dan token usage
2. CSV summary all models dengan average metrics

Usage: python generate_implicit_summary.py
"""
import json
import csv
from pathlib import Path
import re
from collections import defaultdict

def extract_final_answer(response_content):
    """Extract final answer dari response"""
    if not response_content:
        return None
    
    patterns = [
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'final answer.*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'answer is.*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'\$\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_content, re.IGNORECASE)
        if match:
            ans = match.group(1).replace(',', '')
            try:
                if '.' in ans:
                    return float(ans)
                return int(ans)
            except:
                continue
    
    numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', response_content)
    if numbers:
        last_num = numbers[-1].replace(',', '')
        try:
            if '.' in last_num:
                return float(last_num)
            return int(last_num)
        except:
            pass
    
    return None

def extract_ground_truth(ground_truth_text):
    """Extract ground truth number"""
    if not ground_truth_text:
        return None
    
    match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth_text)
    if match:
        ans = match.group(1).replace(',', '')
        try:
            if '.' in ans:
                return float(ans)
            return int(ans)
        except:
            pass
    
    return None

def process_result_file(file_path):
    """Process single result JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_name = data.get('metadata', {}).get('model_info', {}).get('model_id', 'unknown')
        run_id = data.get('metadata', {}).get('run_id', '')
        
        # Parse persona info from run_id
        persona_match = re.search(r'(woman|man)_implicit_(\d+)', run_id)
        persona_type = persona_match.group(1) if persona_match else 'unknown'
        persona_id = int(persona_match.group(2)) if persona_match else 0
        
        question_match = re.search(r'gsm8k_(\d+)', run_id)
        question_id = int(question_match.group(1)) if question_match else 0
        
        # Get response content
        response_content = data.get('response', {}).get('content', '')
        ground_truth_text = data.get('input', {}).get('ground_truth', '')
        
        # Extract answers
        final_answer = extract_final_answer(response_content)
        ground_truth = extract_ground_truth(ground_truth_text)
        
        # Check correctness
        is_correct = None
        if final_answer is not None and ground_truth is not None:
            if isinstance(final_answer, float) or isinstance(ground_truth, float):
                is_correct = abs(float(final_answer) - float(ground_truth)) < 0.01
            else:
                is_correct = int(final_answer) == int(ground_truth)
        
        # Get token usage
        usage = data.get('response', {}).get('raw_api_response', {}).get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        completion_tokens_details = usage.get('completion_tokens_details', {})
        reasoning_tokens = completion_tokens_details.get('reasoning_tokens', 0)
        
        return {
            'model_name': model_name,
            'persona_type': persona_type,
            'persona_id': persona_id,
            'question_id': question_id,
            'is_correct': is_correct,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_per_model_csv(data, output_dir):
    """Generate CSV per model per persona"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group by model and persona
    grouped = defaultdict(lambda: defaultdict(list))
    for item in data:
        grouped[item['model_name']][item['persona_id']].append(item)
    
    csv_count = 0
    for model_name, personas in grouped.items():
        safe_model_name = model_name.replace('/', '_')
        
        for persona_id, items in personas.items():
            persona_type = items[0]['persona_type']
            output_file = Path(output_dir) / f"{safe_model_name}_{persona_type}_implicit_{persona_id}.csv"
            
            # Sort by question_id
            items.sort(key=lambda x: x['question_id'])
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'question_id', 'is_correct', 'prompt_tokens', 'completion_tokens', 
                    'total_tokens', 'reasoning_tokens'
                ])
                writer.writeheader()
                
                for item in items:
                    writer.writerow({
                        'question_id': item['question_id'],
                        'is_correct': 1 if item['is_correct'] else 0 if item['is_correct'] is not None else '',
                        'prompt_tokens': item['prompt_tokens'],
                        'completion_tokens': item['completion_tokens'],
                        'total_tokens': item['total_tokens'],
                        'reasoning_tokens': item['reasoning_tokens'],
                    })
            
            csv_count += 1
            print(f"  [OK] {safe_model_name} {persona_type}_implicit_{persona_id}: {len(items)} rows")
    
    print(f"\n[OK] Generated {csv_count} per-model CSVs")

def generate_summary_csv(data, output_file):
    """Generate summary CSV with accuracy and token metrics per model per persona"""
    # Group by model and persona
    grouped = defaultdict(lambda: defaultdict(list))
    for item in data:
        key = (item['model_name'], item['persona_type'], item['persona_id'])
        grouped[key].append(item)
    
    summaries = []
    for (model_name, persona_type, persona_id), items in grouped.items():
        total = len(items)
        correct = sum(1 for item in items if item['is_correct'] is True)
        incorrect = sum(1 for item in items if item['is_correct'] is False)
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Token metrics
        total_prompt = sum(item['prompt_tokens'] for item in items)
        total_completion = sum(item['completion_tokens'] for item in items)
        total_all = sum(item['total_tokens'] for item in items)
        total_reasoning = sum(item['reasoning_tokens'] for item in items)
        
        avg_prompt = total_prompt / total if total > 0 else 0
        avg_completion = total_completion / total if total > 0 else 0
        avg_total = total_all / total if total > 0 else 0
        avg_reasoning = total_reasoning / total if total > 0 else 0
        
        summaries.append({
            'model_name': model_name,
            'persona_type': persona_type,
            'persona_id': persona_id,
            'total_questions': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy_percent': round(accuracy, 2),
            'total_prompt_tokens': total_prompt,
            'total_completion_tokens': total_completion,
            'total_all_tokens': total_all,
            'total_reasoning_tokens': total_reasoning,
            'avg_prompt_tokens': round(avg_prompt, 2),
            'avg_completion_tokens': round(avg_completion, 2),
            'avg_total_tokens': round(avg_total, 2),
            'avg_reasoning_tokens': round(avg_reasoning, 2),
        })
    
    # Sort by model name and persona
    summaries.sort(key=lambda x: (x['model_name'], x['persona_type'], x['persona_id']))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_name', 'persona_type', 'persona_id', 'total_questions', 'correct', 'incorrect', 
            'accuracy_percent', 'total_prompt_tokens', 'total_completion_tokens', 'total_all_tokens', 
            'total_reasoning_tokens', 'avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens', 
            'avg_reasoning_tokens'
        ])
        writer.writeheader()
        writer.writerows(summaries)
    
    print(f"[OK] Summary CSV: {output_file} ({len(summaries)} rows)")

def main():
    print("="*80)
    print("GENERATING CSV SUMMARY FOR IMPLICIT RESULTS")
    print("="*80)
    print()
    
    # Scan all implicit result files
    patterns = [
        "result/*/woman_implicit_*/gsm8k_*.json",
        "result/*/man_implicit_*/gsm8k_*.json"
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(list(Path(".").glob(pattern)))
    
    print(f"Found {len(all_files)} result files")
    print()
    
    # Process all files
    data = []
    for i, file_path in enumerate(all_files, 1):
        if i % 500 == 0:
            print(f"Processing... {i}/{len(all_files)}")
        
        result = process_result_file(file_path)
        if result:
            data.append(result)
    
    print(f"\n[OK] Processed {len(data)} files successfully")
    print()
    
    # Generate outputs
    output_dir = "analysis/implicit_models"
    summary_file = "analysis/implicit_summary_all.csv"
    
    Path("analysis").mkdir(exist_ok=True)
    
    print("Generating per-model CSVs...")
    generate_per_model_csv(data, output_dir)
    
    print("\nGenerating summary CSV...")
    generate_summary_csv(data, summary_file)
    
    print()
    print("="*80)
    print("DONE!")
    print("="*80)
    print(f"\nOutput:")
    print(f"  - Per-model CSVs: {output_dir}/")
    print(f"  - Summary CSV: {summary_file}")

if __name__ == "__main__":
    main()
