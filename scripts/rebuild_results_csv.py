import sys
import logging
# --- Extraction helpers ---
def get_response_text(resp):
    """Robustly extract response text from various possible fields and formats."""
    # 1. Direct content field
    content = resp.get('content', None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    # 2. Try choices[0].message.content (string or list)
    choices = resp.get('raw_api_response', {}).get('choices', [{}])
    if choices and isinstance(choices, list):
        msg = choices[0].get('message', {})
        # a. content as string
        c = msg.get('content', None)
        if isinstance(c, str) and c.strip():
            return c.strip()
        # b. content as list of parts
        if isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and 'text' in part:
                    parts.append(str(part['text']))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return '\n'.join(parts).strip()
        # c. reasoning/thinking/analysis as fallback
        for key in ['reasoning', 'thinking', 'analysis']:
            val = msg.get(key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
    # 3. Try output_text, response, completion, generated_text, result
    for key in ['output_text', 'response', 'completion', 'generated_text', 'result']:
        val = resp.get(key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # 4. If resp['content'] is a stringified JSON, try to parse
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                for k in ['content','output_text','response','completion','generated_text','result']:
                    v = loaded.get(k, None)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
    # 5. If resp itself is a stringified JSON
    if isinstance(resp, str):
        try:
            loaded = json.loads(resp)
            if isinstance(loaded, dict):
                return get_response_text(loaded)
        except Exception:
            pass
    return ''

# --- NEW IMPLEMENTATION BELOW ---
import os
import re
import csv
import json
import argparse
from collections import Counter
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--report_dir', required=True)
    parser.add_argument('--include_personas', default=None)
    parser.add_argument('--exclude_dirs', default='warmup')
    return parser.parse_args()

PERSONA_LIST = [
    'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
]

FINAL_MARKERS = [
    r'Final Answer:', r'Answer:', r'Therefore', r'So the answer is', r'Thus', r'Hence', r'equals', r'='  # '=' for fallback
]

FINAL_LINE_HINTS = [
    'answer', 'final', 'therefore', 'so', 'thus', 'total', 'equals', '=', 'is'
]

ENUM_LINE_RE = re.compile(r'^\s*(step\s*\d+|\d+[\).:]|-|•)\s', re.I)

# --- Extraction helpers ---
def normalize_number(s):
    s = s.replace(',', '').replace('$', '').strip()
    try:
        return float(s) if '.' in s else int(s)
    except Exception:
        return None

def extract_candidates(text):
    """Return list of (number, start_idx, end_idx, line, context)"""
    candidates = []
    for m in re.finditer(r'([-+]?[$]?[\d,]+(?:\.\d+)?)', text):
        num = normalize_number(m.group(1))
        if num is not None:
            # Find line and context
            start, end = m.span()
            before = text[:start]
            after = text[end:]
            line = before.split('\n')[-1] + m.group(1) + after.split('\n')[0]
            candidates.append((num, start, end, line, m.group(1)))
    return candidates

def extract_boxed(text):
    boxed = re.findall(r'\\boxed\{([^}]*)\}|\\boxed\s*\{([^}]*)\}|\$\\boxed\{([^}]*)\}\$', text)
    boxed_flat = [val for group in boxed[::-1] for val in group if val]
    for val in boxed_flat:
        num = normalize_number(val)
        if num is not None:
            return num, True
    return None, False

def extract_hashes(text):
    m = re.search(r'####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)', text)
    if m:
        num = normalize_number(m.group(1))
        if num is not None:
            return num, True
    return None, False

def extract_marker(text):
    for marker in FINAL_MARKERS:
        m = re.search(marker + r'.{0,20}?([-+]?[$]?[\d,]+(?:\.\d+)?)', text, re.I)
        if m:
            num = normalize_number(m.group(1))
            if num is not None:
                return num, marker
    return None, None

def fallback_scoring(text):
    candidates = extract_candidates(text)
    scores = []
    lines = text.splitlines()
    for idx, (num, start, end, line, raw) in enumerate(candidates):
        score = 0
        lstrip = line.strip().lower()
        # +100 if line has final hint
        if any(hint in lstrip for hint in FINAL_LINE_HINTS):
            score += 100
        # +60 if in notes or last paragraph
        if 'note' in lstrip or 'notes' in lstrip or idx >= len(candidates) - 2:
            score += 60
        # -80 if enumeration/step
        if ENUM_LINE_RE.match(line):
            score -= 80
        # +20 if near end
        if end > len(text) * 0.8:
            score += 20
        scores.append((score, idx, num, raw, line))
    if not scores:
        return None, None
    scores.sort(reverse=True)  # highest score, then last
    return scores[0][2], 'score_fallback'

def extract_answer(text):
    # 1. Boxed
    num, boxed = extract_boxed(text)
    if boxed:
        return num, 'boxed'
    # 2. Hashes
    num, hashes = extract_hashes(text)
    if hashes:
        return num, 'hashes'
    # 3. Marker
    num, marker = extract_marker(text)
    if marker:
        return num, marker
    # 4. Fallback
    num, reason = fallback_scoring(text)
    if num is not None:
        return num, reason
    return None, 'none'

def is_truncated(resp, content, max_tokens, completion_tokens, finish_reason):
    if finish_reason == 'length':
        return True
    if completion_tokens is not None and max_tokens is not None:
        try:
            if int(completion_tokens) >= int(max_tokens) - 5:
                return True
        except Exception:
            pass
    # Heuristic: abrupt ending
    if content:
        last = content.strip()[-50:]
        if not re.search(r'[.!?}\]\)]$', last) and not re.search(r'(boxed|####|Answer:|Final answer:)', last, re.I):
            return True
    return False

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    persona_dirs = [p for p in os.listdir(args.input_root) if p in PERSONA_LIST]
    all_rows = []
    error_counter = Counter()
    persona_counter = Counter()
    for persona in persona_dirs:
        folder = os.path.join(args.input_root, persona)
        files = [f for f in os.listdir(folder) if f.startswith('gsm8k_') and f.endswith('.json')]
        if len(files) != 1319:
            print(f"WARNING: Persona {persona} has {len(files)} JSON files (expected 1319)")
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            with open(fpath, encoding='utf-8') as jf:
                data = json.load(jf)
            inp = data.get('input', {})
            resp = data.get('response', {})
            req = data.get('request', {})
            meta = data.get('metadata', {})
            gt = inp.get('ground_truth', '')
            gt_num = None
            m = re.search(r'####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)', gt)
            if m:
                gt_num = normalize_number(m.group(1))
            qid = inp.get('question_id', '')
            dataset = inp.get('dataset', '')
            # Robust response text extraction
            content = get_response_text(resp)
            # Guardrail: log if completion_tokens > 0 but content is empty
            try:
                ctok = resp.get('raw_api_response', {}).get('usage', {}).get('completion_tokens', 0)
                if (ctok is not None and str(ctok).isdigit() and int(ctok) > 0) and (not content or not content.strip()):
                    # Log warning and snapshot keys for debugging
                    keys_top = list(resp.keys())
                    choices = resp.get('raw_api_response', {}).get('choices', [{}])
                    keys_choices = list(choices[0].keys()) if choices and isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict) else []
                    print(f"WARNING: completion_tokens={ctok} but response_content_full is empty for {persona}/{fname}", file=sys.stderr)
                    print(f"  resp keys: {keys_top}", file=sys.stderr)
                    print(f"  choices[0] keys: {keys_choices}", file=sys.stderr)
            except Exception:
                pass
            extracted, extraction_reason = extract_answer(content)
            is_correct = (extracted == gt_num) if (extracted is not None and gt_num is not None) else False
            has_boxed = '\\boxed' in content
            has_hashes = '####' in content
            boxed_match = has_boxed and (extracted == gt_num) and extraction_reason == 'boxed'
            hashes_match = has_hashes and (extracted == gt_num) and extraction_reason == 'hashes'
            finish_reason = None
            try:
                choices = resp.get('raw_api_response', {}).get('choices', [{}])
                if choices and isinstance(choices, list):
                    finish_reason = choices[0].get('finish_reason', None)
            except Exception:
                pass
            max_tokens = req.get('parameters', {}).get('max_tokens', None)
            completion_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('completion_tokens', None)
            prompt_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('prompt_tokens', None)
            total_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('total_tokens', None)
            reasoning_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('reasoning_tokens', None)
            is_trunc = is_truncated(resp, content, max_tokens, completion_tokens, finish_reason)
            extraction_issue = (extracted is None)
            # Error category
            if is_trunc:
                error_category = 'truncated_output'
            elif extracted is None:
                error_category = 'no_number_found'
            elif has_boxed and extraction_reason != 'boxed':
                error_category = 'picked_intermediate_step_bug'
            elif not is_correct:
                error_category = 'other_extraction_error'
            else:
                error_category = 'ok'
            error_counter[(persona, error_category)] += 1
            persona_counter[persona] += 1
            row = [persona, os.path.relpath(fpath, args.input_root), qid, dataset, gt, gt_num, content, extracted, is_correct, has_boxed, has_hashes, boxed_match, hashes_match, extraction_issue, error_category, extraction_reason, finish_reason, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens]
            all_rows.append(row)
    # Write results.csv
    out_csv = os.path.join(args.report_dir, 'results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(["persona_folder","file","question_id","dataset","ground_truth","ground_truth_answer","response_content_full","extracted_answer","is_correct","has_boxed","has_hashes","boxed_match","hashes_match","extraction_issue","error_category","extraction_reason","finish_reason","prompt_tokens","completion_tokens","reasoning_tokens","total_tokens"])
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {out_csv}")
    print("Error category breakdown:")
    for persona in PERSONA_LIST:
        print(f"{persona:20s}", end=' ')
        for cat in ['ok','truncated_output','no_number_found','picked_intermediate_step_bug','other_extraction_error']:
            print(f"{cat}: {error_counter.get((persona,cat),0):4d}", end=' | ')
        print()

if __name__ == '__main__':
    main()
