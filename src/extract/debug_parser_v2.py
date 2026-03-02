#!/usr/bin/env python3
"""Debug answer parsing - analyze why accuracy is low"""

import json
import os
import re
import sys

# Import the actual parser from generate_analysis_csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_analysis_csv import extract_answer_from_response

def extract_gt(gt):
    if not gt:
        return None
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', gt)
    return float(match.group(1)) if match else None

def has_number_in_content(content, target):
    """Check if target number appears anywhere in content"""
    content_clean = re.sub(r'\{,\}', '', content)
    content_clean = re.sub(r'\\,', '', content_clean)
    all_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', content_clean)
    for n in all_nums:
        try:
            if abs(float(n.replace(',','')) - target) < 0.01:
                return True
        except:
            pass
    return False

# Debug specific models
models = [
    ('openai_gpt-oss-20b', 'man_explicit'),
    ('openai_gpt-oss-120b', 'man_explicit'),
    ('qwen_qwen3-next-80b-a3b-instruct', 'man_explicit'),
    ('groq-llama-3.1-8b-instant', 'man_implicits'),
]

for model, persona in models:
    folder = f'results/{model}/{persona}'
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue
    
    print(f'\n{"="*60}')
    print(f'MODEL: {model}')
    print(f'{"="*60}')
    
    correct = 0
    wrong_extracted = 0  # Extracted but wrong, GT IS in content (parsing bug)
    not_extracted = 0  # Could not extract
    model_wrong = 0  # Model answered wrong (GT not in content)
    
    wrong_examples = []
    not_extracted_examples = []
    
    files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])
    
    for fname in files[:500]:
        filepath = os.path.join(folder, fname)
        with open(filepath, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        gt = extract_gt(data['input'].get('ground_truth',''))
        content = data['response'].get('content','')
        extracted = extract_answer_from_response(content)
        
        if gt is None:
            continue
        
        if extracted is None:
            not_extracted += 1
            if has_number_in_content(content, gt):
                # GT is in content but not extracted - parsing bug
                if len(not_extracted_examples) < 3:
                    not_extracted_examples.append((data['input']['question_id'], gt, content[-500:]))
        elif abs(gt - extracted) < 0.01:
            correct += 1
        else:
            # Extracted wrong number
            if has_number_in_content(content, gt):
                wrong_extracted += 1
                if len(wrong_examples) < 5:
                    wrong_examples.append((data['input']['question_id'], gt, extracted, content[-500:]))
            else:
                model_wrong += 1
    
    total = correct + wrong_extracted + not_extracted + model_wrong
    print(f'\nResults from first 500 questions:')
    print(f'  ✅ Correct: {correct}/{total} ({100*correct/total:.1f}%)')
    print(f'  ❌ Wrong extraction (GT in content - PARSING BUG): {wrong_extracted}')
    print(f'  ⚠️  Not extracted (GT in content - PARSING BUG): {not_extracted}')
    print(f'  🤖 Model wrong (GT not in content): {model_wrong}')
    
    potential_accuracy = (correct + wrong_extracted) / total * 100 if total > 0 else 0
    print(f'\n  📊 Potential accuracy if parsing was perfect: {potential_accuracy:.1f}%')
    
    if wrong_examples:
        print('\n--- PARSING BUG: Wrong extraction examples ---')
        for qid, gt, ext, content in wrong_examples[:2]:
            print(f'\nQ{qid}: GT={gt}, Extracted={ext}')
            print(f'Content:\n{content}')
    
    if not_extracted_examples:
        print('\n--- PARSING BUG: Not extracted examples ---')
        for qid, gt, content in not_extracted_examples[:2]:
            print(f'\nQ{qid}: GT={gt}')
            print(f'Content:\n{content}')
