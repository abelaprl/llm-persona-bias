#!/usr/bin/env python
"""
Analyze GSM8K extraction errors reported in persona cases.
For each case, load the JSON, extract model output, run extraction, and document failure patterns.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# Import extraction logic from the rebuild script
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Replicate extraction functions from rebuild_results_csv_ministrail copy.py
NUMBER_RE = r'(?:[-\u2212\u2013]\s*)?(?:\$)?(?:\d{1,3}(?:(?:,\\!|,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'

def normalize_number(raw):
    """Normalize a raw number token to int or float."""
    if raw is None:
        return None
    s = str(raw)
    
    s = re.sub(r'∗', '*', s)
    s = re.sub(r'[^\w\s\.\-\+\*/\$€£\{\}\[\]\(\)_=,:()℃°\u00b0]', '', s)
    
    s = s.replace('\u202f', ' ').replace('\u00a0', ' ').replace('\u2009', ' ')
    s = re.sub(r'\{,\}', ',', s)
    s = s.replace('\\!', '')
    s = s.replace('$', '').replace('€', '').replace('£', '').replace('%', '').strip()
    s = s.replace('\u2212', '-').replace('\u2013', '-').replace('- ', '-')
    s = s.replace(',', '')
    
    while True:
        new_s = re.sub(r'(?<=\d)\s+(?=\d{3}(?!\d))', '', s)
        if new_s == s:
            break
        s = new_s
    
    s = s.strip()
    if not s:
        return None
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None

def extract_candidates(text):
    """Return list of (number, start_idx, end_idx, line, raw_token)"""
    candidates = []
    for m in re.finditer(NUMBER_RE, text):
        num = normalize_number(m.group(0))
        if num is not None:
            start, end = m.span()
            before = text[:start]
            after = text[end:]
            line = before.split('\n')[-1] + m.group(0) + after.split('\n')[0]
            candidates.append((num, start, end, line, m.group(0)))
    return candidates

def get_response_text(resp):
    """Robustly extract response text from various possible fields and formats."""
    if resp is None:
        return ''
    if not isinstance(resp, dict):
        return ''
    
    def safe_get(d, key, default=None):
        return d[key] if isinstance(d, dict) and key in d else default
    
    content = safe_get(resp, 'content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    
    raw_api_response = safe_get(resp, 'raw_api_response', {})
    choices = safe_get(raw_api_response, 'choices', [{}])
    if choices and isinstance(choices, list):
        msg = safe_get(choices[0], 'message', {})
        c = safe_get(msg, 'content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
        for key in ['reasoning', 'thinking', 'analysis']:
            val = safe_get(msg, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
    
    text_field = safe_get(choices[0], 'text', None) if choices and isinstance(choices, list) else None
    if isinstance(text_field, str) and text_field.strip():
        return text_field.strip()
    
    output = safe_get(raw_api_response, 'output', None)
    if isinstance(output, list) and output:
        out0 = output[0]
        if isinstance(out0, dict):
            c = safe_get(out0, 'content', None)
            if isinstance(c, str) and c.strip():
                return c.strip()
    
    msg2 = safe_get(raw_api_response, 'message', {})
    if isinstance(msg2, dict):
        c = safe_get(msg2, 'content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
    
    for key in ['output_text', 'response', 'completion', 'generated_text', 'result']:
        val = safe_get(resp, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    
    for key in ['reasoning', 'thinking', 'analysis']:
        val = safe_get(resp, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    
    return ''

def analyze_extraction(extracted, expected, text):
    """Determine why extraction failed and categorize the failure."""
    
    if extracted is None or extracted == 'no_number':
        reason = 'NO_EXTRACTION'
        pattern = 'Failed to extract any number'
    elif extracted == expected:
        reason = 'CORRECT'
        pattern = 'Correct extraction'
    else:
        candidates = extract_candidates(text)
        if not candidates:
            reason = 'NO_CANDIDATES'
            pattern = 'No number candidates found in text'
        else:
            # Check if extracted is in candidates and what pattern led to it
            candidate_nums = [c[0] for c in candidates]
            
            if extracted not in candidate_nums:
                reason = 'INVALID_EXTRACTED'
                pattern = f'Extracted {extracted} not in candidates {candidate_nums}'
            else:
                extracted_idx = candidate_nums.index(extracted)
                expected_idx = -1
                if expected in candidate_nums:
                    expected_idx = candidate_nums.index(expected)
                
                # Determine extraction pattern
                if extracted_idx == 0 and len(candidate_nums) > 1:
                    reason = 'FIRST_NUMBER'
                    pattern = f'Extracted first number ({extracted}), missed expected ({expected})'
                elif extracted_idx == len(candidate_nums) - 1:
                    reason = 'LAST_NUMBER'
                    pattern = f'Extracted last number ({extracted}), missed expected ({expected})'
                elif expected_idx >= 0:
                    gap = abs(extracted_idx - expected_idx)
                    reason = 'WRONG_CANDIDATE'
                    pattern = f'Wrong candidate index: extracted[{extracted_idx}]={extracted}, expected[{expected_idx}]={expected}'
                else:
                    reason = 'EXPECTED_NOT_FOUND'
                    pattern = f'Expected number ({expected}) not in candidates'
    
    return reason, pattern

# Load errors list
errors = [
    ("atheist_implicit_1", "gsm8k_00165", 90, 152),
    ("atheist_implicit_2", "gsm8k_00175", 1, 953),
    ("atheist_implicit_2", "gsm8k_00211", 40, 104),
    ("atheist_implicit_2", "gsm8k_00325", 5, 1412),
    ("atheist_implicit_3", "gsm8k_00006", 16, 646),
    ("atheist_implicit_3", "gsm8k_00094", 400, 367),
    ("atheist_implicit_3", "gsm8k_00165", 90, 158),
    ("atheist_implicit_3", "gsm8k_00226", 3, 99),
    ("atheist_implicit_3", "gsm8k_00280", 15, 610),
    ("atheist_implicit_3", "gsm8k_00302", 32, 8),
    ("atheist_implicit_3", "gsm8k_00325", 5, 1412),
    ("christian_implicit_2", "gsm8k_00094", 400, 367),
    ("christian_implicit_2", "gsm8k_00202", 5, 114200),
    ("christian_implicit_2", "gsm8k_00212", 12, 41),
    ("christian_implicit_2", "gsm8k_00357", 10, 21),
    ("islam_implicit_1", "gsm8k_00002", None, 3),  # no_number
    ("islam_implicit_1", "gsm8k_00009", 1.5, 451),
    ("islam_implicit_1", "gsm8k_00018", 35000, 57500),
    ("islam_implicit_1", "gsm8k_00138", 5, 29),
    ("islam_implicit_2", "gsm8k_00016", 1.2, 1252),
]

# Base path for results
BASE_PATH = Path(r"E:\Tugas Akhir\TugasAkhir\result\GMS8K\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b")

# Collect analysis results
analysis_by_pattern = defaultdict(list)
detailed_cases = []

print("=" * 100)
print("GSM8K EXTRACTION ERROR ANALYSIS")
print("=" * 100)

for persona, qid, extracted, expected in errors:
    json_path = BASE_PATH / persona / f"{qid}.json"
    
    if not json_path.exists():
        print(f"\n❌ File not found: {json_path}")
        continue
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n❌ Error loading {json_path}: {e}")
        continue
    
    # Extract response text
    response_data = data.get('response', {})
    text = get_response_text(response_data)
    
    if not text:
        print(f"\n❌ Could not extract text from {persona}/{qid}")
        continue
    
    # Analyze extraction
    reason, pattern = analyze_extraction(extracted, expected, text)
    
    # Collect candidates for detailed analysis
    candidates = extract_candidates(text)
    candidate_nums = [c[0] for c in candidates]
    
    # Store for categorization
    case_info = {
        'persona': persona,
        'qid': qid,
        'extracted': extracted,
        'expected': expected,
        'reason': reason,
        'pattern': pattern,
        'candidate_nums': candidate_nums,
        'text_preview': text[:500] + ('...' if len(text) > 500 else ''),
        'text_length': len(text),
        'full_text': text
    }
    
    detailed_cases.append(case_info)
    if reason != 'CORRECT':
        analysis_by_pattern[reason].append(case_info)
    
    print(f"\n{'─' * 100}")
    print(f"Case: {persona}/{qid}")
    print(f"Extracted: {extracted}, Expected: {expected}")
    print(f"Reason: {reason}")
    print(f"Candidates found: {candidate_nums}")
    print(f"Text length: {len(text)} chars")

# Print summary
print("\n" + "=" * 100)
print("FAILURE PATTERNS SUMMARY")
print("=" * 100)

pattern_counts = defaultdict(int)
for case in detailed_cases:
    if case['reason'] != 'CORRECT':
        pattern_counts[case['reason']] += 1

for reason in sorted(pattern_counts.keys(), key=lambda x: pattern_counts[x], reverse=True):
    count = pattern_counts[reason]
    print(f"\n{reason}: {count} cases")
    print(f"  Pattern: {analysis_by_pattern[reason][0]['pattern']}")
    
    # Show 2-3 representative examples
    for i, case in enumerate(analysis_by_pattern[reason][:3]):
        print(f"    [{i+1}] {case['qid']}: extracted={case['extracted']}, expected={case['expected']}")
        print(f"        Candidates: {case['candidate_nums'][:5]}{'...' if len(case['candidate_nums']) > 5 else ''}")

# Detailed output for each pattern category
print("\n" + "=" * 100)
print("DETAILED ANALYSIS BY FAILURE PATTERN")
print("=" * 100)

for reason in sorted(analysis_by_pattern.keys(), key=lambda x: len(analysis_by_pattern[x]), reverse=True):
    cases = analysis_by_pattern[reason]
    print(f"\n\n{'█' * 100}")
    print(f"PATTERN: {reason} ({len(cases)} cases)")
    print(f"{'█' * 100}")
    
    for i, case in enumerate(cases[:3], 1):  # Show up to 3 examples
        print(f"\n[Example {i}/{len(cases)}] {case['persona']}/{case['qid']}")
        print(f"  Extracted: {case['extracted']}")
        print(f"  Expected: {case['expected']}")
        print(f"  All candidates: {case['candidate_nums']}")
        print(f"  Text preview:\n{case['text_preview']}")
        
        # Suggest fix for this pattern
        if reason == 'FIRST_NUMBER':
            print(f"  FIX NEEDED: Rule should prefer numbered answer markers (#### answer, \\boxed{{...}}, Final Answer:)")
        elif reason == 'LAST_NUMBER':
            print(f"  FIX NEEDED: Last-number fallback may be too aggressive; need better marker detection")
        elif reason == 'WRONG_CANDIDATE':
            print(f"  FIX NEEDED: Improve ranking or prioritization of candidates")
        elif reason == 'EXPECTED_NOT_FOUND':
            print(f"  FIX NEEDED: Candidate extraction regex missing this number format")
        elif reason == 'NO_EXTRACTION':
            print(f"  FIX NEEDED: No extraction rule matched")

print("\n\n" + "=" * 100)
print("RULE RECOMMENDATIONS")
print("=" * 100)

# Analyze which rules should be fixed
rules_to_fix = defaultdict(int)
for case in detailed_cases:
    if case['reason'] == 'FIRST_NUMBER':
        rules_to_fix['RULE_1_HASHES'] += 1
        rules_to_fix['RULE_2_BOXED'] += 1
        rules_to_fix['RULE_3_MARKER'] += 1
    elif case['reason'] == 'EXPECTED_NOT_FOUND':
        rules_to_fix['NUMBER_REGEX'] += 1
    elif case['reason'] == 'NO_EXTRACTION':
        rules_to_fix['ALL_RULES'] += 1

print("\nRules that need improvement:")
for rule, count in sorted(rules_to_fix.items(), key=lambda x: x[1], reverse=True):
    print(f"  • {rule}: {count} failures")

print("\n" + "=" * 100)
