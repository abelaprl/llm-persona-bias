"""
Generate CSV reports organized by model and persona
Output structure: result/reports/{model_name}/{persona_name}/results.csv
Each CSV contains all 1319 GSM8K questions for one model-persona combination

Format baseline:
question_id,persona_folder,persona_type,success,is_correct,ground_truth_answer,extracted_answer,prompt_tokens,completion_tokens,reasoning_tokens,total_tokens,latency_sec,retry_count
"""
import json
import csv
from pathlib import Path
import re

def extract_final_answer(response_content):
    """Extract final answer dari response (angka saja)"""
    if not response_content:
        return None
    
    # Clean up the text - normalize whitespace
    text = response_content.strip()
    
    # Patterns dari yang paling spesifik ke yang lebih umum
    patterns = [
        # LaTeX boxed patterns
        r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}',  # \boxed{18}
        r'\$\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}\$',  # $\boxed{18}$
        r'boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}',  # boxed{18} tanpa backslash
        
        # #### patterns (GSM8K standard)
        r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # #### 18 atau #### $18
        
        # Answer patterns
        r'[Aa]nswer[:\s]+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # Answer: 18
        r'[Aa]nswer\s+is[:\s]+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # Answer is 18
        r'[Ff]inal\s+[Aa]nswer[:\s]+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # Final Answer: 18
        r'[Tt]he\s+answer\s+is[:\s]+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # The answer is 18
        r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:x\s*=\s*)?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # Therefore, x = 5
        
        # Bold/markdown patterns
        r'\*\*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*',  # **$18**
        r'\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*(?:\s*(?:dollars?|eggs?|tomatoes?|cars?|minutes?|hours?|\$|\.)?)?$',  # **18** at end
        
        # Equals pattern at end
        r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|tomatoes?|cars?)?\.?\s*$',  # = 18 at end
        
        # So/Thus conclusion patterns
        r'[Ss]o[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|\.)?\.?\s*$',
        r'[Tt]hus[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Remove commas and convert to number
            ans = match.group(1).replace(',', '').strip()
            try:
                return float(ans)
            except:
                continue
    
    # Fallback: ambil angka terakhir di text (hanya angka standalone, bukan dalam kata)
    # Cari pattern angka yang lebih strict - harus diikuti/didahului spasi/punctuation
    numbers = re.findall(r'(?:^|[\s=:$])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)(?:[\s\.$,;:!?]|$)', text)
    if numbers:
        last_num = numbers[-1].replace(',', '').strip()
        try:
            return float(last_num)
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
            return float(ans)
        except:
            pass
    
    return None

def process_result_file(file_path):
    """Process single result JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract question_id from filename: gsm8k_00001.json -> 1
        filename = file_path.stem  # gsm8k_00001
        match = re.search(r'gsm8k_(\d+)', filename)
        question_id = int(match.group(1)) if match else 0
        
        # Extract info from metadata
        model_info = data.get('metadata', {}).get('model_info', {})
        model_name = model_info.get('model_id', 'unknown')
        
        # Extract from input
        input_data = data.get('input', {})
        
        # Get persona info
        persona_data = input_data.get('persona', {})
        persona_type_full = persona_data.get('type', 'unknown')  # e.g., "man_implicit"
        persona_id = persona_data.get('id', 0)
        
        # persona_folder = man_implicit_1, woman_implicit_2, etc.
        persona_folder = f"{persona_type_full}_{persona_id}"
        
        # Get response data
        response_data = data.get('response', {})
        raw_response = response_data.get('raw_api_response', {})
        
        # Get response content - try multiple sources
        response_content = response_data.get('content', '')
        
        # If content is empty, try to get from raw_api_response choices
        if not response_content:
            choices = raw_response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                response_content = message.get('content', '')
                
                # If still empty, try reasoning field (some models put answer there)
                if not response_content:
                    reasoning = message.get('reasoning', '')
                    if reasoning:
                        response_content = reasoning
        
        # Get ground truth from input
        ground_truth_text = input_data.get('ground_truth', '')
        
        # Extract final answers
        final_answer = extract_final_answer(response_content)
        ground_truth = extract_ground_truth(ground_truth_text)
        
        # Check correctness
        is_correct = None
        if final_answer is not None and ground_truth is not None:
            # Handle float comparison with tolerance
            is_correct = abs(float(final_answer) - float(ground_truth)) < 0.01
        
        # Get token usage from raw_api_response (already defined above)
        usage = raw_response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Reasoning tokens (might not exist for all models)
        completion_tokens_details = usage.get('completion_tokens_details', {})
        reasoning_tokens = completion_tokens_details.get('reasoning_tokens', 0) if completion_tokens_details else 0
        
        # Get latency from performance
        performance = response_data.get('performance', {})
        latency_sec = performance.get('latency_sec', 0)
        
        # Get retry count from performance (default 0 if not present)
        retry_count = performance.get('retry_count', 0)
        
        # success = both answers extracted successfully
        success = final_answer is not None and ground_truth is not None
        
        return {
            'question_id': question_id,
            'persona_folder': persona_folder,
            'persona_type': persona_type_full,
            'success': success,
            'is_correct': is_correct,
            'ground_truth_answer': ground_truth,
            'extracted_answer': final_answer,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': total_tokens,
            'latency_sec': round(latency_sec, 3) if latency_sec else 0,
            'retry_count': retry_count,
        }
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return None

def generate_csv_for_model_persona(model_folder, persona_folder, output_path):
    """Generate CSV for one model-persona combination"""
    
    # Get all JSON files in the persona folder
    json_files = list(persona_folder.glob('gsm8k_*.json'))
    
    if not json_files:
        print(f"  [SKIP] No JSON files found in {persona_folder}")
        return 0
    
    # Process all JSON files
    results = []
    for json_file in json_files:
        result = process_result_file(json_file)
        if result:
            results.append(result)
    
    if not results:
        print(f"  [SKIP] No valid results in {persona_folder}")
        return 0
    
    # Sort by question_id
    results.sort(key=lambda x: x['question_id'])
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV - format sesuai baseline
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
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    return len(results)

def main():
    print("="*80)
    print("GENERATING REPORTS BY MODEL AND PERSONA")
    print("="*80)
    print()
    
    # Base directories
    result_dir = Path("result")
    reports_dir = result_dir / "reports"
    
    # Get all model folders (exclude testing and reports)
    model_folders = [f for f in result_dir.iterdir() 
                     if f.is_dir() and f.name not in ['testing', 'reports']]
    
    if not model_folders:
        print("No model folders found in result/")
        return
    
    print(f"Found {len(model_folders)} model folders")
    print()
    
    total_csv_count = 0
    total_questions = 0
    
    # Process each model
    for model_folder in sorted(model_folders):
        model_name = model_folder.name
        print(f"Processing model: {model_name}")
        
        # Get all persona folders in this model (only implicit ones)
        persona_folders = [f for f in model_folder.iterdir() 
                          if f.is_dir() and 'implicit' in f.name]
        
        if not persona_folders:
            print(f"  [SKIP] No persona folders found")
            continue
        
        # Process each persona
        for persona_folder in sorted(persona_folders):
            persona_name = persona_folder.name
            
            # Output path: result/reports/model_name/persona_name/results.csv
            output_path = reports_dir / model_name / persona_name / "results.csv"
            
            # Generate CSV
            count = generate_csv_for_model_persona(model_folder, persona_folder, output_path)
            
            if count > 0:
                print(f"  [OK] {persona_name}: {count} questions -> {output_path}")
                total_csv_count += 1
                total_questions += count
            
        print()
    
    print("="*80)
    print(f"DONE! Generated {total_csv_count} CSV files")
    print(f"Total questions processed: {total_questions}")
    print(f"Output directory: {reports_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()
