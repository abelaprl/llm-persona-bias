#!/usr/bin/env python3
"""Quick analysis of answer extraction accuracy with no-match samples."""
import json, re
from pathlib import Path

def extract_gold(text):
    if not text: return None
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    return float(m.group(1).replace(',','')) if m else None

def extract_model(text):
    if not text: return None, 'empty', ''
    patterns = [
        # Standard patterns
        (r'[Ff]inal\s*[Aa]nswer[:\s]*\*?\*?\s*\$?\s*([-]?[\d,]+(?:\.\d+)?)', 'std'),
        (r'\*\*[Ff]inal\s*[Aa]nswer[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)\*\*', 'bold1'),
        (r'[Ff]inal\s*[Aa]nswer[:\s]*\*\*([-]?[\d,]+(?:\.\d+)?)\*\*', 'bold2'),
        (r'\*\*[Ff]inal\s*[Aa]nswer:?\*\*[:\s]*\$?\s*([-]?[\d,]+(?:\.\d+)?)', 'bold3'),
        # Bold label with colon: **Final answer**: X
        (r'\*\*[Ff]inal\s*[Aa]nswer\*\*[:\s]*\$?€?\s*([-]?[\d,]+(?:\.\d+)?)', 'bold_label'),
        # LaTeX formats
        (r'\\boxed\{([-]?[\d,]+(?:\.\d+)?)\}', 'boxed'),
        (r'\\boxed\{\\dfrac\{(\d+)\}\{\d+\}\}', 'boxed_frac'),  # Take numerator for fractions
        (r'\\mathbf\{([-]?[\d,]+(?:\.\d+)?)\}', 'mathbf'),
        # LaTeX with comma separator: \(14,\!400\)
        (r'\\\(([-]?[\d,\\!]+(?:\.\d+)?)\\\)', 'latex_paren'),
        # Euro/currency: €77 or **€77**
        (r'[:\s]\*?\*?€\s*([-]?[\d,]+(?:\.\d+)?)\*?\*?', 'euro'),
        # <number> placeholder: Final answer: <number> 48
        (r'[Ff]inal\s*[Aa]nswer[:\s]*<number>\s*([-]?[\d,]+(?:\.\d+)?)', 'placeholder'),
        # Hash format
        (r'####\s*([-]?[\d,]+(?:\.\d+)?)', 'hash'),
        # Pattern: = $XXX at end of text (like "= $144")
        (r'=\s*\$?([-]?[\d,]+(?:\.\d+)?)\s*\.?\s*$', 'equals_end'),
        # Pattern: calculation result in LaTeX like \( 40 - 10 = 30 \)
        (r'=\s*([-]?[\d,]+(?:\.\d+)?)\s*\\\)', 'latex_result'),
        # Pattern: totals X or cost is X at end
        (r'(?:total|cost|is|equals?)\s+(?:is\s+)?\$?([-]?[\d,]+(?:\.\d+)?)\s*\.?\s*$', 'total_end'),
    ]
    for p, n in patterns:
        matches = list(re.finditer(p, text))
        if matches:
            try: 
                val = matches[-1].group(1).replace(',','').replace('\\!','')
                return float(val), n, matches[-1].group(0)[:60]
            except: pass
    return None, 'no_match', text[-200:] if len(text)>200 else text

models = {
    'bert_nebulon': ['man_implicits','woman_implicits'], 
    'grok_4_1': ['man_implicits','woman_implicits'], 
    'nemotron-nano': ['man_implicits','woman_implicits']
}

print('Model                Persona            Total  Correct  Wrong  NoMatch  Accuracy')
print('-'*85)

all_no_match = []
for model, personas in models.items():
    for persona in personas:
        path = Path('results') / model / persona / 'gsm8k_raw'
        if not path.exists(): 
            print(f'{model:<20} {persona:<18} NOT FOUND')
            continue
        files = sorted(path.glob('gsm8k_*.json'))
        correct = wrong = no_match = 0
        for f in files:
            try:
                d = json.load(open(f, encoding='utf-8'))
                resp = d.get('response_text','')
                if not resp:
                    ch = d.get('response',{}).get('choices',[])
                    if ch: resp = ch[0].get('message',{}).get('content','')
                gold = extract_gold(d.get('input',{}).get('gold_answer',''))
                model_ans, pat, dbg = extract_model(resp)
                if gold is None: continue
                if model_ans is None:
                    no_match += 1
                    if len(all_no_match) < 30:
                        all_no_match.append((model, persona, f.name, dbg))
                elif abs(model_ans - gold) < 0.01:
                    correct += 1
                else:
                    wrong += 1
            except Exception as e: 
                pass
        total = correct + wrong + no_match
        acc = correct/total*100 if total else 0
        print(f'{model:<20} {persona:<18} {total:<6} {correct:<8} {wrong:<6} {no_match:<8} {acc:.2f}%')

print()
print('='*85)
print('SAMPLES WITH NO MATCH (need pattern fix):')
print('='*85)
for m, p, fn, dbg in all_no_match:
    print(f'\n{m}/{p}/{fn}:')
    # Clean up for display
    dbg_clean = dbg.replace('\n', ' ').strip()
    print(f'  ...{dbg_clean[-150:]}')
