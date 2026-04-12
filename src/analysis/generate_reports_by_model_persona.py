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
import sys
import argparse


def _last_numeric_token_as_int(text: str):
    """Return last numeric token in text, drop fractional part and thousands separators."""
    if not text:
        return None
    try:
        matches = re.findall(r'([+-]?\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', text)
        if not matches:
            return None
        s = matches[-1]
        if '.' in s and ',' in s:
            s = s.replace(',', '')
            s = s.split('.')[0]
        elif '.' in s:
            s = s.split('.')[0]
        elif ',' in s:
            parts = s.split(',')
            if len(parts[-1]) == 3:
                s = s.replace(',', '')
            else:
                s = ','.join(parts[:-1]) if len(parts) > 1 else parts[0]
                s = s.replace(',', '')
        s = re.sub(r'[^0-9+-]', '', s)
        return float(s)
    except:
        return None

def extract_final_answer(response_content):
    """Extract final answer dari response - improved multi-priority extraction"""
    if not response_content:
        return None
    
    text = response_content.strip()
    
    # Priority 1: boxed pattern - handle with/without %, \text{}, etc., take LAST match
    boxed_matches = list(re.finditer(r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[^}]*\}', text))
    if boxed_matches:
        ans = boxed_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 2: #### pattern - take LAST match
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 3: Look for "Answer" line and extract carefully
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines) >= 15 else lines
    
    for line in reversed(last_lines):
        line_stripped = line.strip()
        
        # Check if this is an Answer line
        if re.search(r'^\*?\*?[Aa]nswer', line_stripped) or re.search(r'\*\*[Aa]nswer', line_stripped):
            # Pattern 1: "Answer: $NUMBER" or "**Answer: $NUMBER**"
            match = re.search(r'[Aa]nswer[:\s\*]*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 2: "Answer: NUMBER" directly after colon  
            match = re.search(r'[Aa]nswer[:\s\*]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 3: "have NUMBER unit" in answer line
            match = re.search(r'\bhave\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sheep|miles?|dollars?|eggs?|items?|points?|downloads?|pages?|books?|apples?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 4: "is NUMBER unit" - get NUMBER before unit  
            match = re.search(r'\bis\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:miles?|dollars?|eggs?|vacuum\s*cleaners?|glasses?|items?|people|students?|apples?|books?|pages?|points?|pieces?|downloads?|minutes?|sheep)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 5: "pay $NUMBER"
            match = re.search(r'\bpay\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 6: Bold number **NUMBER** in answer line (any position)
            bold_matches = re.findall(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', line_stripped)
            if bold_matches:
                return float(bold_matches[-1].replace(',', ''))
            
            # Pattern 7: "give (her chickens)? NUMBER cups/unit" - for chicken feed problem
            match = re.search(r'\bgive\s+(?:her\s+)?(?:chickens?\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|bolts?|items?|pieces?|dollars?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 8: "needs to give NUMBER" 
            match = re.search(r'\bneeds?\s+to\s+give\s+(?:her\s+)?(?:chickens?\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|bolts?|items?|pieces?)\b', line_stripped, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 9: Generic - find LAST "NUMBER unit" in Answer line (fallback)
            all_unit_matches = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|items?|dollars?|miles?|eggs?|pieces?|minutes?|hours?|days?|years?|meters?|bolts?|sheep|glasses?|vacuum\s*cleaners?|downloads?|pages?|books?|people|students?)\b', line_stripped, re.IGNORECASE)
            if all_unit_matches:
                return float(all_unit_matches[-1].replace(',', ''))
    
    # Priority 4: Look for "The answer is NUMBER"
    for line in reversed(last_lines):
        match = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 5: Bold number **NUMBER unit** anywhere in last 5 lines
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|sheep|miles?(?:\s+per\s+hour)?|dollars?|minutes?|downloads?|eggs?|pages?|books?|items?|pieces?|hours?|days?|meters?|dozens?|years?|bolts?|%|percent)?\*\*', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 6: "makes/made/profit of $NUMBER" / "speed of NUMBER" patterns
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:makes?|made|profit\s+of|speed\s+of)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 7: "needs NUMBER" / "have NUMBER" / "takes NUMBER" patterns 
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:needs?|have|has|gets?|total\s+of|runs?\s+a\s+total\s+of|run|eats?|will\s+eat|takes?|pay\s+a\s+total\s+of|it\s+takes)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?|dollars?|minutes?|downloads?|eggs?|pages?|books?|items?|pieces?|meters?|dozens?|bolts?|years?)?\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 8: "is NUMBER unit" with extended units
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\bis\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?(?:\s+per\s+hour)?|dollars?|minutes?|downloads?|eggs?|pages?|books?|items?|pieces?|glasses?|points?|people|students?|meters?|dozens?)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 9: = NUMBER unit (not just at end)
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|downloads?|sheep|meters?|cups?|bolts?|minutes?|years?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 10: = NUMBER at end of line  
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 11: "give NUMBER" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\bgive\s+(?:her\s+)?(?:chickens?\s+)?\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|bolts?|items?|pieces?)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 12: "distance/salary/profit/earnings is/are $NUMBER" patterns
    for line in reversed(last_lines[-5:]):
        match = re.search(r'(?:distance|salary|profit|earnings?|would\s+be)\s+(?:is|are|of)?\s*\*?\*?\\?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 13: "covers NUMBER miles/distance" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\bcovers?\s+(?:a\s+)?(?:total\s+)?(?:distance\s+of\s+)?\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:miles?|meters?|km)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 14: "NUMBER%" pattern for percentages 
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 15: "Answer: \(NUMBER\)" LaTeX inline math
    for line in reversed(last_lines[-5:]):
        match = re.search(r'[Aa]nswer[:\s]*\\\(([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\\\)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 16: "= NUMBER m." or "= NUMBER m" at end of line
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:m|meters?)\.?\s*\*?\*?\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 17: "NUMBER\ \text{min}" LaTeX format
    for line in reversed(last_lines[-5:]):
        match = re.search(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?\s*\\text\{', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 18: escaped dollar \$NUMBER pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\\?\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[.\s]*(?:$|[^0-9])', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 19: "began with **NUMBER" pattern
    for line in reversed(last_lines[-5:]):
        match = re.search(r'began\s+with\s+\*?\*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Fallback: take last numeric token in entire response and drop fractional part
    try:
        num_matches = re.findall(r'([+-]?\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)', text)
        if num_matches:
            s = num_matches[-1]
            if '.' in s and ',' in s:
                s = s.replace(',', '')
                s = s.split('.')[0]
            elif '.' in s:
                s = s.split('.')[0]
            elif ',' in s:
                parts = s.split(',')
                if len(parts[-1]) == 3:
                    s = s.replace(',', '')
                else:
                    s = ','.join(parts[:-1]) if len(parts) > 1 else parts[0]
                    s = s.replace(',', '')
            s = re.sub(r'[^0-9+-]', '', s)
            return float(s)
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
        # Override with last numeric token (drop fractional part) to avoid picking intermediate '=' matches
        last_num = _last_numeric_token_as_int(response_content)
        if last_num is not None:
            final_answer = last_num
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
            'response_content': response_content,
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
        'response_content',
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
    parser = argparse.ArgumentParser(description='Generate CSV reports by model and persona')
    parser.add_argument('--model', type=int, help='Model index (1-based). Run without args to see list.')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--force', action='store_true', help='Overwrite existing CSVs if present')
    args = parser.parse_args()
    
    # Base directories
    result_dir = Path("result")
    reports_dir = result_dir / "reports"
    
    # Get all model folders (exclude testing and reports)
    model_folders = sorted([f for f in result_dir.iterdir() 
                     if f.is_dir() and f.name not in ['testing', 'reports']])
    
    if not model_folders:
        print("No model folders found in result/", flush=True)
        return
    
    # Show list of models
    if args.list or args.model is None:
        print("="*80, flush=True)
        print("AVAILABLE MODELS", flush=True)
        print("="*80, flush=True)
        for i, folder in enumerate(model_folders, 1):
            print(f"  {i:2d}. {folder.name}", flush=True)
        print()
        print("Usage: python script.py --model <number>", flush=True)
        print("Example: python script.py --model 1", flush=True)
        print("="*80, flush=True)
        return
    
    # Validate model index
    model_idx = args.model
    if model_idx < 1 or model_idx > len(model_folders):
        print(f"Error: Model index must be between 1 and {len(model_folders)}", flush=True)
        return
    
    # Process single model
    model_folder = model_folders[model_idx - 1]
    model_name = model_folder.name
    
    print("="*80, flush=True)
    print(f"PROCESSING MODEL {model_idx}: {model_name}", flush=True)
    print("="*80, flush=True)
    print(flush=True)
    
    total_csv_count = 0
    total_questions = 0
    
    # Get all persona folders in this model (implicit AND explicit)
    persona_folders = [f for f in model_folder.iterdir() 
                      if f.is_dir() and ('implicit' in f.name or 'explicit' in f.name)]
    
    if not persona_folders:
        print(f"[SKIP] No persona folders found", flush=True)
        return
    
    # Process each persona
    for persona_folder in sorted(persona_folders):
        persona_name = persona_folder.name
        
        # Output path: result/reports/model_name/persona_name/results.csv
        output_path = reports_dir / model_name / persona_name / "results.csv"
        
        # Skip if already exists unless --force
        if output_path.exists() and not args.force:
            print(f"  [SKIP] {persona_name}: already exists", flush=True)
            continue
        
        # Generate CSV
        count = generate_csv_for_model_persona(model_folder, persona_folder, output_path)
        
        if count > 0:
            print(f"  [OK] {persona_name}: {count} questions -> {output_path}", flush=True)
            total_csv_count += 1
            total_questions += count
    
    print(flush=True)
    print("="*80, flush=True)
    print(f"DONE! Model {model_idx} ({model_name})", flush=True)
    print(f"Generated {total_csv_count} CSV files, {total_questions} questions", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
