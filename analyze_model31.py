"""Comprehensive analysis of Model 31 (allenai_olmo-3-32b-think)"""
import json
import re
import os
from collections import defaultdict

def extract_final_answer(response_content):
    if not response_content:
        return None
    
    text = response_content.strip()
    # normalize unicode narrow/no-break spaces to normal spaces
    text = text.replace('\u2009', ' ').replace('\u202f', ' ').replace('\xa0', ' ')

    # helper to parse a numeric token (allow $ and grouping commas/spaces)
    def parse_number_token(s):
        if not s:
            return None
        s = s.replace('\u2009', '').replace('\u202f', '').replace('\xa0', '')
        s = s.replace(',', '').replace(' ', '')
        s = s.replace('$', '')
        try:
            return float(s)
        except:
            return None

    # Priority 1: boxed pattern (allow \boxed{...} inside [] or ())
    boxed_any = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_any:
        # prefer last boxed numeric inside boxed text
        for content in reversed(boxed_any):
            m = re.search(r'([+-]?\$?\s*[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', content)
            if m:
                val = parse_number_token(m.group(1))
                if val is not None:
                    return val
    
    # Priority 2: #### pattern
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1)
        val = parse_number_token(ans)
        if val is not None:
            return val
    
    # Priority 3: Look for "Answer" line
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines) >= 15 else lines
    
    for line in reversed(last_lines):
        line_stripped = line.strip()
        
        if re.search(r'^\*?\*?[Aa]nswer', line_stripped) or re.search(r'\*\*[Aa]nswer', line_stripped):
            match = re.search(r'[Aa]nswer[:\s\*]*\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', line_stripped)
            if match:
                val = parse_number_token(match.group(1))
                if val is not None:
                    return val
            
            match = re.search(r'[Aa]nswer[:\s\*]+([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', line_stripped)
            if match:
                val = parse_number_token(match.group(1))
                if val is not None:
                    return val
            
            bold_matches = re.findall(r'\*\*\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)\*\*', line_stripped)
            if bold_matches:
                val = parse_number_token(bold_matches[-1])
                if val is not None:
                    return val
    
    # Priority 4: "The answer is NUMBER"
    for line in reversed(last_lines):
        match = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', line)
        if match:
            val = parse_number_token(match.group(1))
            if val is not None:
                return val
    
    # Priority 5: Bold number **NUMBER**
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\*\*\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|days?|meters?|dozens?|years?|bolts?|feet|foot|centimeters?|cm|inches?|%|percent)?\*\*', line, re.IGNORECASE)
        if match:
            val = parse_number_token(match.group(1))
            if val is not None:
                return val
    
    # Priority 6: "is/are NUMBER unit"
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:is|are)\s+\*?\*?\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)\*?\*?\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|glasses?|points?|people|students?|meters?|dozens?|yards?|burritos?|pounds?|lb|leftover)\b', line, re.IGNORECASE)
        if match:
            val = parse_number_token(match.group(1))
            if val is not None:
                return val
    
    # Priority 7: = NUMBER at end
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)\s*$', line, re.IGNORECASE)
        if match:
            val = parse_number_token(match.group(1))
            if val is not None:
                return val

    # Final fallback: last numeric token in the final lines, but avoid obvious intermediate tokens
    final_candidates = []
    for line in reversed(last_lines):
        for m in re.finditer(r'([+-]?\$?\s*[0-9][0-9,\s\u2009\u202f]*\.?[0-9]*)', line):
            token = m.group(1)
            # skip if preceding context suggests unit/subtotal/intermediate
            pre = line[:m.start()].lower()
            bad_context = any(w in pre for w in ['per', 'each', 'price', 'cost', 'subtotal', 'per day', 'per week', 'per month', 'rate', 'unit', 'hour', 'minute', 'pair', 'apiece', 'each.'])
            if not bad_context:
                val = parse_number_token(token)
                if val is not None:
                    final_candidates.append((val, token, line))
    if final_candidates:
        # choose last reasonable candidate
        return final_candidates[0][0]
    
    return None

def extract_ground_truth(ground_truth_text):
    if not ground_truth_text:
        return None
    match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth_text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None

def is_truncated(text):
    """Check if response appears truncated"""
    if not text:
        return True, "EMPTY"
    text = text.strip()
    if not text:
        return True, "EMPTY"
    endings = '.!?)]}"\'0123456789'
    if text[-1] not in endings:
        return True, f"ENDS_WITH: '{text[-20:]}'"
    return False, "OK"

# Analyze model 31
model = 'allenai_olmo-3-32b-think'
model_path = f'result/{model}'

stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'no_extract': 0, 'empty': 0, 'truncated': 0, 'total': 0, 'length_limit': 0})
all_issues = []

print(f"=" * 80)
print(f"COMPREHENSIVE ANALYSIS: {model}")
print(f"=" * 80)

for persona in sorted(os.listdir(model_path)):
    persona_path = os.path.join(model_path, persona)
    if not os.path.isdir(persona_path):
        continue
    
    files = sorted([f for f in os.listdir(persona_path) if f.endswith('.json')])
    
    for fname in files:
        fpath = os.path.join(persona_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data.get('response', {}).get('content', '')
            gt_text = data.get('input', {}).get('ground_truth', '')
            
            # Try to get reasoning if content is empty (for thinking models)
            reasoning = ''
            if not content:
                raw = data.get('response', {}).get('raw_api_response', {})
                choices = raw.get('choices', [])
                if choices:
                    msg = choices[0].get('message', {})
                    reasoning = msg.get('reasoning', '')
            
            # Check finish_reason
            raw = data.get('response', {}).get('raw_api_response', {})
            choices = raw.get('choices', [])
            finish_reason = ''
            if choices:
                finish_reason = choices[0].get('finish_reason', '')
            
            gt_val = extract_ground_truth(gt_text)
            extracted = extract_final_answer(content)
            is_trunc, trunc_info = is_truncated(content)
            
            stats[persona]['total'] += 1
            
            # Track finish_reason = length (hit token limit)
            if finish_reason == 'length':
                stats[persona]['length_limit'] += 1
            
            if not content or content.strip() == '':
                stats[persona]['empty'] += 1
                # Try to extract from reasoning field
                extracted_from_reasoning = extract_final_answer(reasoning) if reasoning else None
                if extracted_from_reasoning is not None:
                    if extracted_from_reasoning == gt_val:
                        stats[persona]['correct'] += 1
                    else:
                        stats[persona]['wrong'] += 1
                else:
                    stats[persona]['no_extract'] += 1
                    all_issues.append({
                        'persona': persona,
                        'file': fname,
                        'issue': 'EMPTY' + ('+LENGTH_LIMIT' if finish_reason == 'length' else ''),
                        'gt': gt_val,
                        'extracted': None,
                        'detail': f'Content empty, finish_reason={finish_reason}, reasoning_len={len(reasoning)}'
                    })
            elif is_trunc:
                stats[persona]['truncated'] += 1
                if extracted is None:
                    stats[persona]['no_extract'] += 1
                    all_issues.append({
                        'persona': persona,
                        'file': fname,
                        'issue': 'TRUNCATED+NO_EXTRACT',
                        'gt': gt_val,
                        'extracted': None,
                        'detail': trunc_info
                    })
                elif extracted == gt_val:
                    stats[persona]['correct'] += 1
                else:
                    stats[persona]['wrong'] += 1
            elif extracted is None:
                stats[persona]['no_extract'] += 1
                all_issues.append({
                    'persona': persona,
                    'file': fname,
                    'issue': 'NO_EXTRACT',
                    'gt': gt_val,
                    'extracted': None,
                    'detail': content[-150:] if content else ''
                })
            elif extracted == gt_val:
                stats[persona]['correct'] += 1
            else:
                stats[persona]['wrong'] += 1
                if len([i for i in all_issues if i['issue'] == 'WRONG']) < 10:
                    all_issues.append({
                        'persona': persona,
                        'file': fname,
                        'issue': 'WRONG',
                        'gt': gt_val,
                        'extracted': extracted,
                        'detail': content[-150:] if content else ''
                    })
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

# Print summary per persona
print(f"\n{'Persona':<20} {'Total':<8} {'Correct':<10} {'Wrong':<8} {'NoExt':<8} {'Empty':<8} {'LenLim':<8} {'Trunc':<8} {'Acc':<8}")
print("-" * 90)

total_all = {'correct': 0, 'wrong': 0, 'no_extract': 0, 'empty': 0, 'truncated': 0, 'total': 0, 'length_limit': 0}
for persona in sorted(stats.keys()):
    s = stats[persona]
    acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
    print(f"{persona:<20} {s['total']:<8} {s['correct']:<10} {s['wrong']:<8} {s['no_extract']:<8} {s['empty']:<8} {s['length_limit']:<8} {s['truncated']:<8} {acc:.1f}%")
    for k in total_all:
        total_all[k] += s[k]

print("-" * 90)
acc_total = total_all['correct'] / total_all['total'] * 100 if total_all['total'] > 0 else 0
print(f"{'TOTAL':<20} {total_all['total']:<8} {total_all['correct']:<10} {total_all['wrong']:<8} {total_all['no_extract']:<8} {total_all['empty']:<8} {total_all['length_limit']:<8} {total_all['truncated']:<8} {acc_total:.1f}%")

# Print sample issues
print(f"\n" + "=" * 80)
print("SAMPLE ISSUES (first few of each type):")
print("=" * 80)

# Group issues by type
issues_by_type = defaultdict(list)
for issue in all_issues:
    issues_by_type[issue['issue']].append(issue)

for issue_type, issues in issues_by_type.items():
    print(f"\n### {issue_type} ({len(issues)} total) ###")
    for i, issue in enumerate(issues[:3]):
        print(f"\n{i+1}. {issue['persona']}/{issue['file']}")
        print(f"   GT: {issue['gt']} | Extracted: {issue['extracted']}")
        print(f"   Detail: {repr(issue['detail'][:100])}")
