"""
Generate CSV dari hasil GSM8K experiment
Output: 
1. CSV detail per soal per model
2. CSV summary per model (rata-rata + total)
"""
import json
import csv
from pathlib import Path
import re
from collections import defaultdict

def extract_final_answer(response_content):
    """Extract final answer dari response (angka saja)"""
    if not response_content:
        return None
    
    # Cari pattern #### atau final answer
    patterns = [
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'final answer.*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'answer is.*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'\$\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_content, re.IGNORECASE)
        if match:
            # Remove commas and convert to number
            ans = match.group(1).replace(',', '')
            try:
                # Try as int first, then float
                if '.' in ans:
                    return float(ans)
                return int(ans)
            except:
                continue
    
    # Fallback: ambil angka terakhir di text
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
    """Extract ground truth number dari text"""
    if not ground_truth_text:
        return None
    
    # Cari pattern #### 
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
        
        # Extract info
        model_name = data.get('model', {}).get('model_id', 'unknown')
        run_id = data.get('run_id', '')
        
        # Parse run_id untuk get persona info
        # Format: {model}_{persona_type}_implicit_{persona_id}_gsm8k_{question_id}
        persona_match = re.search(r'(woman|man)_implicit_(\d+)', run_id)
        persona_type = persona_match.group(1) if persona_match else 'unknown'
        persona_id = int(persona_match.group(2)) if persona_match else 0
        
        question_match = re.search(r'gsm8k_(\d+)', run_id)
        question_id = int(question_match.group(1)) if question_match else 0
        
        # Get response content
        response_content = data.get('response', {}).get('content', '')
        
        # Get ground truth
        ground_truth_text = data.get('ground_truth', {}).get('answer', '')
        
        # Extract final answers
        final_answer = extract_final_answer(response_content)
        ground_truth = extract_ground_truth(ground_truth_text)
        
        # Check correctness
        is_correct = None
        if final_answer is not None and ground_truth is not None:
            # Handle float comparison with tolerance
            if isinstance(final_answer, float) or isinstance(ground_truth, float):
                is_correct = abs(float(final_answer) - float(ground_truth)) < 0.01
            else:
                is_correct = int(final_answer) == int(ground_truth)
        
        # Get token usage
        usage = data.get('response', {}).get('raw_api_response', {}).get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Reasoning tokens (might not exist for all models)
        completion_tokens_details = usage.get('completion_tokens_details', {})
        reasoning_tokens = completion_tokens_details.get('reasoning_tokens', 0)
        
        # Get latency
        latency_sec = data.get('response', {}).get('latency', 0)
        
        # Get retry count (default 0 if not present)
        retry_count = data.get('retry_count', 0)
        
        return {
            'model_name': model_name,
            'persona_type': persona_type,
            'persona_id': persona_id,
            'question_id': question_id,
            'final_answer': final_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens,
            'latency_sec': round(latency_sec, 3),
            'retry_count': retry_count,
            'response_preview': response_content[:100] if response_content else '',
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_detailed_csv(results, output_file, persona_type):
    """Generate detailed CSV per question"""
    if not results:
        print(f"No results to write for {persona_type}")
        return
    
    fieldnames = [
        'question_id',
        'model_name',
        'persona_folder',
        'persona_type',
        'success',
        'is_correct',
        'ground_truth_answer',
        'extracted_answer',
        'prompt_tokens',
        'completion_tokens',
        'reasoning_tokens',
        'total_tokens',
        'latency_sec',
        'retry_count'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            persona_folder = f"{persona_type}_implicit_{result['persona_id']}"
            success = result['final_answer'] is not None and result['ground_truth'] is not None
            
            writer.writerow({
                'question_id': result['question_id'],
                'model_name': result['model_name'],
                'persona_folder': persona_folder,
                'persona_type': f"{persona_type}_implicit",
                'success': success,
                'is_correct': result['is_correct'] if result['is_correct'] is not None else '',
                'ground_truth_answer': result['ground_truth'] if result['ground_truth'] is not None else '',
                'extracted_answer': result['final_answer'] if result['final_answer'] is not None else '',
                'prompt_tokens': result['prompt_tokens'],
                'completion_tokens': result['completion_tokens'],
                'reasoning_tokens': result['reasoning_tokens'],
                'total_tokens': result['total_tokens'],
                'latency_sec': result.get('latency_sec', 0),
                'retry_count': result.get('retry_count', 0),
            })
    
    print(f"[OK] Detailed CSV saved: {output_file}")

def generate_summary_csv(results, output_file, persona_type):
    """Generate summary CSV per model with aggregations"""
    if not results:
        print(f"No results to summarize for {persona_type}")
        return
    
    # Group by model and persona
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        key = (result['model_name'], result['persona_id'])
        grouped[key]['results'].append(result)
    
    # Calculate aggregations
    summaries = []
    for (model_name, persona_id), data in grouped.items():
        results_list = data['results']
        
        # Count correct/incorrect
        correct_count = sum(1 for r in results_list if r['is_correct'] is True)
        incorrect_count = sum(1 for r in results_list if r['is_correct'] is False)
        total_questions = len(results_list)
        
        # Accuracy
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # Token aggregations
        total_prompt_tokens = sum(r['prompt_tokens'] for r in results_list)
        total_completion_tokens = sum(r['completion_tokens'] for r in results_list)
        total_all_tokens = sum(r['total_tokens'] for r in results_list)
        total_reasoning_tokens = sum(r['reasoning_tokens'] for r in results_list)
        
        avg_prompt_tokens = total_prompt_tokens / total_questions if total_questions > 0 else 0
        avg_completion_tokens = total_completion_tokens / total_questions if total_questions > 0 else 0
        avg_total_tokens = total_all_tokens / total_questions if total_questions > 0 else 0
        avg_reasoning_tokens = total_reasoning_tokens / total_questions if total_questions > 0 else 0
        
        summaries.append({
            'model_name': model_name,
            'persona_id': persona_id,
            'total_questions': total_questions,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'accuracy_percent': round(accuracy, 2),
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_all_tokens': total_all_tokens,
            'total_reasoning_tokens': total_reasoning_tokens,
            'avg_prompt_tokens': round(avg_prompt_tokens, 2),
            'avg_completion_tokens': round(avg_completion_tokens, 2),
            'avg_total_tokens': round(avg_total_tokens, 2),
            'avg_reasoning_tokens': round(avg_reasoning_tokens, 2),
        })
    
    # Sort by model name and persona id
    summaries.sort(key=lambda x: (x['model_name'], x['persona_id']))
    
    # Write CSV
    fieldnames = [
        'model_name',
        'persona_id',
        'total_questions',
        'correct_count',
        'incorrect_count',
        'accuracy_percent',
        'total_prompt_tokens',
        'total_completion_tokens',
        'total_all_tokens',
        'total_reasoning_tokens',
        'avg_prompt_tokens',
        'avg_completion_tokens',
        'avg_total_tokens',
        'avg_reasoning_tokens'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    
    print(f"[OK] Summary CSV saved: {output_file}")
    print(f"     Total models: {len(summaries)}")

def generate_per_model_csv(results, output_dir, persona_type):
    """Generate separate CSV for each model and each persona"""
    if not results:
        print(f"No results for per-model CSV")
        return
    
    # Group by model and persona
    grouped = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[result['model_name']][result['persona_id']].append(result)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    csv_count = 0
    # Generate CSV for each model+persona combination
    for model_name, personas in grouped.items():
        # Safe filename
        safe_model_name = model_name.replace('/', '_')
        
        for persona_id, persona_results in personas.items():
            # Sort by question_id
            persona_results.sort(key=lambda x: x['question_id'])
            
            output_file = Path(output_dir) / f"model_{safe_model_name}_{persona_type}_implicit_{persona_id}.csv"
            
            fieldnames = [
                'question_id',
                'persona_folder',
                'persona_type',
                'success',
                'is_correct',
                'ground_truth_answer',
                'extracted_answer',
                'prompt_tokens',
                'completion_tokens',
                'reasoning_tokens',
                'total_tokens',
                'latency_sec',
                'retry_count'
            ]
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in persona_results:
                    persona_folder = f"{persona_type}_implicit_{result['persona_id']}"
                    success = result['final_answer'] is not None and result['ground_truth'] is not None
                    
                    writer.writerow({
                        'question_id': result['question_id'],
                        'persona_folder': persona_folder,
                        'persona_type': f"{persona_type}_implicit",
                        'success': success,
                        'is_correct': result['is_correct'] if result['is_correct'] is not None else '',
                        'ground_truth_answer': result['ground_truth'] if result['ground_truth'] is not None else '',
                        'extracted_answer': result['final_answer'] if result['final_answer'] is not None else '',
                        'prompt_tokens': result['prompt_tokens'],
                        'completion_tokens': result['completion_tokens'],
                        'reasoning_tokens': result['reasoning_tokens'],
                        'total_tokens': result['total_tokens'],
                        'latency_sec': result.get('latency_sec', 0),
                        'retry_count': result.get('retry_count', 0),
                    })
            
            csv_count += 1
            print(f"  [OK] {safe_model_name} persona {persona_id}: {len(persona_results)} rows")
    
    print(f"\n[OK] Generated {csv_count} CSV files in: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate CSV from GSM8K results')
    parser.add_argument('--persona_type', type=str, choices=['woman', 'man'], required=True, 
                        help='Persona type: woman or man')
    args = parser.parse_args()
    
    persona_type = args.persona_type
    
    print("="*80)
    print(f"GENERATING CSV FOR: {persona_type.upper()} IMPLICIT")
    print("="*80)
    print()
    
    # Scan all result files
    pattern = f"result/*/{persona_type}_implicit_*/gsm8k_*.json"
    files = list(Path(".").glob(pattern))
    
    print(f"Found {len(files)} result files")
    print()
    
    # Process all files
    results = []
    for i, file_path in enumerate(files, 1):
        if i % 100 == 0:
            print(f"Processing... {i}/{len(files)}")
        
        result = process_result_file(file_path)
        if result:
            results.append(result)
    
    print(f"\n[OK] Processed {len(results)} files successfully")
    print()
    
    # Generate CSVs
    detailed_csv = f"analysis/{persona_type}_implicit_detailed.csv"
    summary_csv = f"analysis/{persona_type}_implicit_summary.csv"
    per_model_dir = f"analysis/{persona_type}_implicit_models"
    
    # Create output directory
    Path("analysis").mkdir(exist_ok=True)
    
    print("Generating detailed CSV...")
    generate_detailed_csv(results, detailed_csv, persona_type)
    
    print("\nGenerating summary CSV...")
    generate_summary_csv(results, summary_csv, persona_type)
    
    print("\nGenerating per-model CSVs...")
    generate_per_model_csv(results, per_model_dir, persona_type)
    
    print()
    print("="*80)
    print("DONE!")
    print("="*80)

if __name__ == "__main__":
    main()
