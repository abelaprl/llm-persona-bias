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

# Check llama responses that failed extraction
model = 'meta-llama_llama-3.1-8b-instruct'
model_path = f'result/{model}'
no_extract = []
random.seed(42)

for persona in os.listdir(model_path)[:2]:
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
            
            if extract_final_answer(content) is None and content:
                no_extract.append((fname, gt_val, content[-300:]))
        except Exception as e:
            print(f"Error: {e}")
    if len(no_extract) >= 6:
        break

print('=' * 60)
print('LLAMA-3.1-8B: NoExtract samples (response endings)')
print('=' * 60)
for i, (fname, gt, ending) in enumerate(no_extract[:6]):
    print(f'\n--- {i+1}. {fname} (GT: {gt}) ---')
    print(ending)
