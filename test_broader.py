"""Broader test of extraction accuracy"""
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from generate_reports_by_model_persona import extract_final_answer

def get_gt(gt_text):
    match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', gt_text)
    return float(match.group(1).replace(',','')) if match else None

# Test multiple models
models_to_test = [
    'google_gemini-2.5-flash',
    'deepseek_deepseek-v3.2', 
    'allenai_olmo-3-32b-think',
    'meta-llama_llama-3.3-70b-instruct'
]

print("="*60)
print("BROADER EXTRACTION TEST")
print("="*60)

for model_name in models_to_test:
    persona_dir = Path(f'result/{model_name}/man_implicit_1')
    if not persona_dir.exists():
        print(f"\n[SKIP] {model_name} - not found")
        continue
    
    files = sorted(persona_dir.glob('gsm8k_*.json'))[:20]
    
    correct = 0
    total = 0
    mismatches = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        # Get response content
        response = data.get('response', {})
        content = response.get('content', '')
        if not content:
            raw = response.get('raw_api_response', {})
            choices = raw.get('choices', [])
            if choices:
                msg = choices[0].get('message', {})
                content = msg.get('content', '') or msg.get('reasoning', '')
        
        # Extract
        extracted = extract_final_answer(content)
        gt = get_gt(data.get('input', {}).get('ground_truth', ''))
        
        total += 1
        if extracted is not None and gt is not None:
            if abs(extracted - gt) < 0.01:
                correct += 1
            else:
                mismatches.append((f.name, gt, extracted))
    
    pct = 100*correct/total if total > 0 else 0
    print(f"\n{model_name}: {correct}/{total} ({pct:.1f}%)")
    if mismatches[:3]:
        for name, gt, ext in mismatches[:3]:
            print(f"  Mismatch: {name} GT={gt}, Ext={ext}")

print("\n" + "="*60)
