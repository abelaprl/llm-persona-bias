"""Comprehensive analysis for openai/gpt-oss-120b (model 34)
Duplicate of analyze_model31 extractor logic but targeted at model 34 results.
"""
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
    endings = '.!?)]}\"\'0123456789'
    if text[-1] not in endings:
        return True, f"ENDS_WITH: '{text[-20:]}'"
    return False, "OK"

# Analyze model 34
model = 'openai_gpt-oss-120b'
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
                if len([i for i in all_issues if i['issue'] == 'WRONG']) < 30:
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

# Write report files for downstream inspection
try:
    report = {
        'model': model,
        'stats': {k: dict(v) for k, v in stats.items()},
        'total': total_all,
        'issues': all_issues,
    }
    with open('analyze34_report.json', 'w', encoding='utf-8') as outf:
        json.dump(report, outf, ensure_ascii=False, indent=2)

    # top-30 mismatches (human-friendly)
    top30 = all_issues[:30]
    with open('analyze34_mismatches_top30.txt', 'w', encoding='utf-8') as m:
        for it in top30:
            m.write(f"{it.get('persona')}/{it.get('file')}\tGT:{it.get('gt')}\tEX:{it.get('extracted')}\n")

    with open('analyze34_mismatches.json', 'w', encoding='utf-8') as jf:
        json.dump(all_issues, jf, ensure_ascii=False, indent=2)

    print('\nWrote analyze34_report.json, analyze34_mismatches.json, analyze34_mismatches_top30.txt')
except Exception as e:
    print(f'Failed writing report files: {e}')
"""Comprehensive analysis of Model 34 (openai_gpt-oss-120b)"""
import json
import re
import os
from collections import defaultdict

def extract_final_answer(response_content):
    if not response_content:
        return None
    
    text = response_content.strip()
    
    # Priority 1: boxed pattern
    boxed_matches = list(re.finditer(r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)[^}]*\}', text))
    if boxed_matches:
        ans = boxed_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 2: #### pattern
    hash_matches = list(re.finditer(r'####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text))
    if hash_matches:
        ans = hash_matches[-1].group(1).replace(',', '')
        try:
            return float(ans)
        except:
            pass
    
    # Priority 3: Look for "Answer" line
    lines = text.strip().split('\n')
    last_lines = lines[-15:] if len(lines) >= 15 else lines
    
    for line in reversed(last_lines):
        line_stripped = line.strip()
        
        if re.search(r'^\*?\*?[Aa]nswer', line_stripped) or re.search(r'\*\*[Aa]nswer', line_stripped):
            match = re.search(r'[Aa]nswer[:\s\*]*\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            match = re.search(r'[Aa]nswer[:\s\*]+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line_stripped)
            if match:
                return float(match.group(1).replace(',', ''))
            
            bold_matches = re.findall(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', line_stripped)
            if bold_matches:
                return float(bold_matches[-1].replace(',', ''))
    
    # Priority 4: "The answer is NUMBER"
    for line in reversed(last_lines):
        match = re.search(r'[Tt]he\s+answer\s+is[:\s]*\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 5: Bold number **NUMBER**
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\*\*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|days?|meters?|dozens?|years?|bolts?|feet|foot|centimeters?|cm|inches?|%|percent)?\*\*', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 6: "is/are NUMBER unit"
    for line in reversed(last_lines[-5:]):
        match = re.search(r'\b(?:is|are)\s+\*?\*?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*?\*?\s*(?:cups?|sheep|miles?|dollars?|minutes?|hours?|downloads?|eggs?|pages?|books?|items?|pieces?|glasses?|points?|people|students?|meters?|dozens?|yards?|burritos?|pounds?|lb|leftover)\b', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Priority 7: = NUMBER at end
    for line in reversed(last_lines[-5:]):
        match = re.search(r'=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))

    # Priority 8: Bold or dollar-wrapped numbers anywhere (fallback)
    bold_dollar = re.findall(r'\*\*\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\*\*', text)
    if bold_dollar:
        try:
            return float(bold_dollar[-1].replace(',', ''))
        except:
            pass

    # Priority 9: Last standalone number in the text (last-resort fallback)
    last_num = None
    for m in re.finditer(r'\$?([+-]?\d{1,3}(?:[\d,]*\d)?(?:\.\d+)?)', text):
        last_num = m.group(1)
    if last_num:
        try:
            return float(last_num.replace(',', ''))
        except:
            pass

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
    # Normalize trailing markdown/LaTeX wrappers that don't indicate truncation
    # Remove trailing asterisks, backticks, dollar signs, ellipses and unmatched brackets
    stripped = re.sub(r'[\*`\$\s]+$', '', text)
    stripped = re.sub(r'\\\]|\\\)$', '', stripped)
    if not stripped:
        return True, "EMPTY_AFTER_STRIP"

    last_char = stripped[-1]
    # Consider OK if ends with typical sentence terminator or a digit (final answer)
    ok_endings = set('.!?)\]\"\'0123456789%')
    if last_char in ok_endings:
        return False, "OK"

    # If text ends with common LaTeX or markdown tokens but has a final number earlier, treat as OK
    if re.search(r'\$?\d+(?:,\d{3})*(?:\.\d+)?\s*\*\*?\s*$', text):
        return False, "OK_NUM_WITH_MARKUP"

    return True, f"ENDS_WITH: '{text[-30:]}'"

# Analyze model 34
model = 'openai_gpt-oss-120b'
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
