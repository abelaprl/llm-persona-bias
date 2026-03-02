#!/usr/bin/env python3
"""Debug answer parsing - find cases where extraction fails"""

import json
import os
import re

def extract_gt(gt):
    if not gt:
        return None
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', gt)
    return float(match.group(1)) if match else None

def extract_answer_old(content):
    """Old simple extraction"""
    if not content:
        return None
    patterns = [
        r'\\boxed\{([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\}',
        r'answer\s*(?:is|:)\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'\$\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.findall(p, content, re.IGNORECASE)
        if m:
            return float(m[-1].replace(',',''))
    return None

# Check models
models = [
    ('openai_gpt-oss-120b', 'man_explicit'),
    ('openai_gpt-oss-20b', 'man_explicit'),
    ('qwen_qwen3-next-80b-a3b-thinking', 'man_explicit'),
    ('qwen_qwen3-next-80b-a3b-instruct', 'man_explicit'),
    ('groq-llama-3.1-8b-instant', 'man_implicits'),
]

print("=== ANALYZING WRONG EXTRACTIONS ===\n")

wrong_samples = []

for model, persona in models:
    folder = f'results/{model}/{persona}'
    if not os.path.exists(folder):
        continue
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")
    
    files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])
    
    wrong_count = 0
    for fname in files[:200]:  # Check first 200
        filepath = os.path.join(folder, fname)
        with open(filepath, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        gt = extract_gt(data['input'].get('ground_truth', ''))
        content = data['response'].get('content', '')
        extracted = extract_answer_old(content)
        
        if gt is not None and extracted is not None and abs(gt - extracted) > 0.01:
            qid = data['input']['question_id']
            wrong_count += 1
            
            if wrong_count <= 5:  # Show first 5 wrong examples
                print(f"\n--- Question {qid} ---")
                print(f"Ground Truth: {gt}")
                print(f"Extracted: {extracted}")
                print(f"Content (last 400 chars):")
                print(content[-400:])
                wrong_samples.append({
                    'model': model,
                    'qid': qid,
                    'gt': gt,
                    'extracted': extracted,
                    'content_end': content[-400:]
                })
    
    print(f"\nTotal wrong in first 200: {wrong_count}")

# Analyze patterns
print("\n\n" + "="*60)
print("PATTERN ANALYSIS")
print("="*60)

# Collect unique patterns at end of responses
patterns_found = {}
for sample in wrong_samples:
    content = sample['content_end']
    # Look for the last line/section pattern
    lines = content.strip().split('\n')
    last_lines = '\n'.join(lines[-5:]) if len(lines) >= 5 else content
    
    if '\\boxed' in last_lines:
        # Extract what's inside boxed
        match = re.search(r'\\boxed\{([^}]+)\}', last_lines)
        if match:
            boxed_content = match.group(1)
            key = f"boxed: {boxed_content[:50]}"
            patterns_found[key] = patterns_found.get(key, 0) + 1

print("\nBoxed patterns found:")
for k, v in sorted(patterns_found.items(), key=lambda x: -x[1]):
    print(f"  {v}x: {k}")
