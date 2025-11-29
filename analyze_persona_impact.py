#!/usr/bin/env python3
"""
Analyze pengaruh persona terhadap akurasi LLM pada GSM8K
Compare akurasi: dengan persona vs tanpa persona
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_answer(response_text):
    """Extract final answer from model response"""
    if not response_text:
        return None
    patterns = [
        r'[Ff]inal\s+answer\s*:\s*\*?\*?(-?\d+(?:\.\d+)?)\*?\*?',
        r'###\s*(\d+(?:\.\d+)?)',
        r'\*\*(-?\d+(?:\.\d+)?)\*\*'
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    return None

def load_test_data():
    """Load GSM8K test data"""
    with open('data/gsm8k_test.json') as f:
        data = json.load(f)
        # Create mapping: Q_id -> expected_answer
        test_map = {}
        for idx, q in enumerate(data):
            test_map[idx] = q.get('answer')  # or use some other id field
        return test_map, data

def analyze_model_responses(model_dir, model_name, test_data):
    """Analyze responses from a model"""
    files = sorted(Path(model_dir).glob("gsm8k_*.json"))
    
    results = {
        'total_files': len(files),
        'correct': 0,
        'total': 0,
        'accuracy': 0,
        'by_range': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for f in files:
        try:
            q_id = int(f.stem.replace('gsm8k_', ''))
        except:
            continue
            
        if q_id >= len(test_data):
            continue
            
        try:
            with open(f) as fp:
                data = json.load(fp)
                response = data.get('response', '')
                expected = test_data[q_id].get('answer')
                
                if not expected:
                    continue
                    
                extracted = extract_answer(response)
                if extracted is not None:
                    exp_num = float(str(expected).strip())
                    results['total'] += 1
                    
                    if abs(extracted - exp_num) < 0.001:
                        results['correct'] += 1
                    
                    # Track by range
                    if q_id < 300:
                        range_key = 'Q0-Q300'
                    elif q_id < 600:
                        range_key = 'Q300-Q600'
                    elif q_id < 900:
                        range_key = 'Q600-Q900'
                    else:
                        range_key = 'Q900+'
                    
                    results['by_range'][range_key]['total'] += 1
                    if abs(extracted - exp_num) < 0.001:
                        results['by_range'][range_key]['correct'] += 1
        except Exception as e:
            continue
    
    if results['total'] > 0:
        results['accuracy'] = 100 * results['correct'] / results['total']
    
    return results

print("="*80)
print("ANALISIS PENGARUH PERSONA TERHADAP AKURASI")
print("="*80)
print()

# Load test data
test_data, raw_data = load_test_data()
print(f"Total test questions: {len(test_data)}\n")

# Analyze current data
models = [
    ("results/grok_4_1_fast/man_implicits/gsm8k_raw", "GROK (man_implicits)"),
    ("results/bert_nebulon_alpha/man_implicits/gsm8k_raw", "NEBULON (man_implicits)"),
]

for model_dir, model_name in models:
    results = analyze_model_responses(model_dir, model_name, test_data)
    
    print(f"\n{model_name}:")
    print(f"  {'─'*60}")
    print(f"  Total files: {results['total_files']}")
    print(f"  Processable: {results['total']}")
    print(f"  Correct: {results['correct']}")
    print(f"  **Overall Accuracy: {results['accuracy']:.2f}%**")
    print()
    
    # By range
    if results['by_range']:
        print(f"  Accuracy by question range:")
        for range_key in sorted(results['by_range'].keys()):
            r = results['by_range'][range_key]
            if r['total'] > 0:
                acc = 100 * r['correct'] / r['total']
                print(f"    {range_key}: {acc:.2f}% ({r['correct']}/{r['total']})")
    print()

print("\n" + "="*80)
print("INTERPRETASI:")
print("="*80)
print("""
Jika akurasi antara different ranges (Q0-Q300 vs Q300+) berbeda signifikan:
  ➜ Persona BERPENGARUH pada performa model
  
Jika akurasi konsisten across ranges:
  ➜ Persona TIDAK berpengaruh OR efek sudah terdistribusi merata

Catatan: Data Q0-Q300 masih incomplete karena API error.
Next step: Retry Q0-Q300 dengan API key yang valid.
""")
