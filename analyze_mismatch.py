import json
import random
import re
import os

def extract_final_answer(text):
    if not text or not isinstance(text, str):
        return None
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        value = boxed_match.group(1)
        numbers = re.findall(r'-?[\d,]+\.?\d*', value.replace(',', ''))
        if numbers:
            return numbers[-1].replace(',', '')
    hash_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if hash_match:
        return hash_match.group(1).replace(',', '')
    answer_bold = re.search(r'\*\*[Aa]nswer[:\*]*\*?\*?\s*[:\s]*\*?\*?.*?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if answer_bold:
        return answer_bold.group(1).replace(',', '')
    answer_colon = re.search(r'(?:The\s+)?[Aa]nswer(?:\s+is)?[:\s=]+\s*\$?(-?[\d,]+\.?\d*)', text)
    if answer_colon:
        return answer_colon.group(1).replace(',', '')
    lines = text.strip().split('\n')
    for line in reversed(lines[-10:]):
        line_clean = line.strip()
        fallback = re.search(r'(?:is|are|=|:)\s*\$?\s*\*?\*?(-?[\d,]+\.?\d*)\s*(?:\*\*)?\.?\s*$', line_clean)
        if fallback:
            return fallback.group(1).replace(',', '')
    for line in reversed(lines[-10:]):
        line_clean = line.strip()
        match = re.search(r'\*\*\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*\*\*', line_clean)
        if match:
            return match.group(1).replace(',', '')
    return None

def is_truncated(text):
    """Check if response appears truncated"""
    if not text:
        return True
    text = text.strip()
    if not text:
        return True
    # Check if ends mid-word/sentence (not ending with proper punctuation or digit)
    endings = '.!?)]}"\'0123456789'
    if text[-1] not in endings:
        return True
    return False

# Analyze multiple models
models_to_check = [
    ('meta-llama_llama-3.1-8b-instruct', 'llama-3.1-8b'),
    ('mistralai_ministral-8b-2512', 'ministral-8b'),
    ('google_gemma-2-9b-it', 'gemma-2-9b'),
    ('google_gemini-2.5-flash', 'gemini-2.5'),
    ('allenai_olmo-3-32b-think', 'olmo-3-32b'),
    ('deepseek_deepseek-v3.2', 'deepseek')
]

random.seed(42)
print("=" * 80)
print(f"{'Model':<18} {'Total':<8} {'Correct':<10} {'NoExt':<8} {'Trunc':<8} {'Wrong':<8} {'Acc':<8}")
print("=" * 80)

for model_dir, model_name in models_to_check:
    model_path = f'result/{model_dir}'
    if not os.path.exists(model_path):
        continue
    
    correct = 0
    no_extract = 0
    truncated = 0
    wrong = 0
    total = 0
    
    for persona in os.listdir(model_path)[:4]:
        persona_path = os.path.join(model_path, persona)
        if not os.path.isdir(persona_path):
            continue
        files = [f for f in os.listdir(persona_path) if f.endswith('.json')]
        sampled = random.sample(files, min(50, len(files)))
        
        for fname in sampled:
            fpath = os.path.join(persona_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = data.get('response', {}).get('content', '')
                gt = data.get('input', {}).get('ground_truth', '')
                gt_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', gt)
                gt_val = gt_match.group(1).replace(',', '') if gt_match else None
                total += 1
                
                is_trunc = is_truncated(content)
                extracted = extract_final_answer(content)
                
                if is_trunc:
                    truncated += 1
                
                if extracted is None:
                    no_extract += 1
                elif extracted == gt_val:
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                pass
    
    acc = correct/total*100 if total > 0 else 0
    print(f"{model_name:<18} {total:<8} {correct:<10} {no_extract:<8} {truncated:<8} {wrong:<8} {acc:.1f}%")

print("=" * 80)
print("\nLegend:")
print("  Correct  = Extracted matches ground truth")
print("  NoExt    = Could not extract answer from response")
print("  Trunc    = Response appears truncated (incomplete)")
print("  Wrong    = Extracted a number but it's wrong (model error)")
print("\nNote: Trunc and NoExt/Wrong can overlap")
