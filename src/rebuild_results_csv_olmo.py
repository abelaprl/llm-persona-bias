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
    # 2b. Try choices[0]["text"] (sometimes used by some APIs)
    if choices and isinstance(choices, list):
        text_field = choices[0].get('text', None)
        if isinstance(text_field, str) and text_field.strip():
            return text_field.strip()
    # 2c. Try raw_api_response["output"][0]["content"] (list/dict)
    output = resp.get('raw_api_response', {}).get('output', None)
    if isinstance(output, list) and output:
        out0 = output[0]
        if isinstance(out0, dict):
            c = out0.get('content', None)
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
    # 2d. Try raw_api_response["message"]["content"]
    msg2 = resp.get('raw_api_response', {}).get('message', {})
    if isinstance(msg2, dict):
        c = msg2.get('content', None)
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
    # 6. Fallback: reasoning/thinking/analysis at top level
    for key in ['reasoning', 'thinking', 'analysis']:
        val = resp.get(key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ''

# --- NEW IMPLEMENTATION BELOW ---
import os
import re
import csv
import json
import argparse
import math
from collections import Counter
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--report_dir', required=True)
    parser.add_argument('--model_slug', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--include_personas', default=None)
    parser.add_argument('--exclude_dirs', default='warmup')
    return parser.parse_args()

def sanitize_model_slug(name):
    if not name:
        return 'unknown_model'
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', name.strip())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or 'unknown_model'

def default_model_name_from_slug(slug):
    return re.sub(r'[_-]+', ' ', slug).strip().title()

def to_number(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    sval = str(val).strip()
    if not sval:
        return 0.0
    try:
        return float(sval)
    except Exception:
        return 0.0

PERSONA_LIST = [
    'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'neutral',
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


# --- Improved extraction logic ---
def extract_final_answer(text):
    """Extract final answer with robust priority and fallback."""
    if not text or not isinstance(text, str):
        return None, 'none'
    tail = text[-1200:] if len(text) > 1200 else text
    lines = tail.splitlines()[-15:]  # last 15 lines
    # 1. #### <number> in tail
    m = re.search(r'####\s*(-?[$]?[\d,]+(?:\.\d+)?)', tail)
    if m:
        num = normalize_number(m.group(1))
        if num is not None:
            return num, 'hashes'
    # 2. \boxed{...} in tail
    boxed = re.findall(r'\\boxed\{([^}]*)\}|\\boxed\s*\{([^}]*)\}|\$\\boxed\{([^}]*)\}\$', tail)
    boxed_flat = [val for group in boxed[::-1] for val in group if val]
    for val in boxed_flat:
        num = normalize_number(val)
        if num is not None:
            return num, 'boxed'
    # 3. Final marker lines (Answer:, Final:, etc) in last lines
    for line in reversed(lines):
        if re.search(r'(Final Answer|Answer|So the answer is|Therefore|Thus|Hence)', line, re.I):
            nums = list(re.finditer(r'(-?[$]?[\d,]+(?:\.\d+)?)', line))
            if nums:
                # Take last number on the line
                num = normalize_number(nums[-1].group(1))
                if num is not None:
                    return num, 'final_marker'
    # 4. Equation/sentence tail lines in last lines (including truncated text)
    for line in reversed(lines):
        l = line.strip()
        if not l:
            continue
        nums = list(re.finditer(r'(-?[$]?[\d,]+(?:\.\d+)?)', l))
        if not nums:
            continue
        has_answer_hint = bool(re.search(r'(answer|therefore|thus|hence|will be|is|so)', l, re.I))
        has_math_hint = ('=' in l) or ('\\frac' in l)
        if has_answer_hint or has_math_hint:
            num = normalize_number(nums[-1].group(1))
            if num is not None:
                return num, 'tail_equation'
    # 5. Bold/boxed (also allow truncated opening) and guarded currency in last lines
    for line in reversed(lines):
        # Complete bold number
        m = re.search(r'\*\*\s*(-?[$]?[\d,]+(?:\.\d+)?)\s*\*\*', line)
        if m:
            num = normalize_number(m.group(1))
            if num is not None:
                return num, 'bold_final'
        # Truncated bold answer like "**... 26" without closing **
        m = re.search(r'\*\*[^*\n]*?(-?[$]?[\d,]+(?:\.\d+)?)\s*$', line)
        if m:
            num = normalize_number(m.group(1))
            if num is not None:
                return num, 'bold_truncated'
        # Truncated boxed like "\\boxed{10" near end
        m = re.search(r'\\boxed\{\s*(-?[$]?[\d,]+(?:\.\d+)?)\s*$', line)
        if m:
            num = normalize_number(m.group(1))
            if num is not None:
                return num, 'boxed_truncated'
        # Currency fallback only when line looks answer-like
        if not re.search(r'(answer|therefore|thus|hence|total|pay|cost|will be|is|=)', line, re.I):
            continue
        m = re.search(r'[$]([\d,]+(?:\.\d+)?)', line)
        if m:
            num = normalize_number(m.group(1))
            if num is not None:
                return num, 'currency_final'
    # 6. Fallback: last number in output, skip percent/ratio/enumeration/if/note
    all_lines = text.splitlines()
    for idx in range(len(all_lines)-1, -1, -1):
        line = all_lines[idx].strip()
        if not line:
            continue
        # Skip lines that are clearly not answer
        if re.match(r'^(Step \d+|\d+[).:]|[-•])', line):
            continue
        if re.match(r'^(If|Assumption|Note)', line, re.I):
            continue
        nums = list(re.finditer(r'(-?[$]?[\d,]+(?:\.\d+)?)', line))
        # Skip percent/ratio
        nums = [m for m in nums if not (m.end() < len(line) and (line[m.end():m.end()+2].startswith('%') or line[m.end():m.end()+2].startswith('/')))]
        if nums:
            num = normalize_number(nums[-1].group(1))
            if num is not None:
                return num, 'last_number_fallback'
    return None, 'none'


# --- Improved truncated detection ---
def has_final_marker(text):
    if not text or not isinstance(text, str):
        return False
    tail = text[-1200:] if len(text) > 1200 else text
    if re.search(r'####\s*-?[$]?[\d,]+', tail):
        return True
    if re.search(r'\\boxed\{[^}]+\}', tail):
        return True
    if re.search(r'(Answer|Final|Therefore|Thus|Hence|So the answer is)', tail, re.I):
        # Accept if line with marker has number
        for line in tail.splitlines()[-15:]:
            if re.search(r'(Answer|Final|Therefore|Thus|Hence|So the answer is)', line, re.I):
                if re.search(r'(-?[$]?[\d,]+(?:\.\d+)?)', line):
                    return True
    # Bold number in tail
    if re.search(r'\*\*\s*-?[$]?[\d,]+(?:\.\d+)?\s*\*\*', tail):
        return True
    return False

def is_truly_truncated(text, finish_reason, completion_tokens, max_tokens):
    # If finish_reason != length, not truncated
    if finish_reason != 'length':
        return False
    # If has final marker, not truncated
    if has_final_marker(text):
        return False
    # Check for open sentence/unfinished output
    if not text or not isinstance(text, str):
        return True
    tail = text.strip()[-120:]
    # Unbalanced brackets
    if tail.count('{') > tail.count('}') or tail.count('[') > tail.count(']') or tail.count('(') > tail.count(')'):
        return True
    # Ends with open math or incomplete
    if re.search(r'[=:(,\[\{]$|\\\[$', tail):
        return True
    # Ends with incomplete word or abrupt stop
    if not re.search(r'[.!?}\]\)]$', tail):
        return True
    return False

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    inferred_slug = os.path.basename(os.path.normpath(args.input_root))
    model_slug = sanitize_model_slug(args.model_slug or inferred_slug)
    model_name = args.model_name or default_model_name_from_slug(model_slug)
    persona_dirs = [p for p in os.listdir(args.input_root) if p in PERSONA_LIST]
    all_rows = []
    error_counter = Counter()
    persona_counter = Counter()
    correct_counter = Counter()
    total_counter = Counter()
    prompt_sum_counter = Counter()
    completion_sum_counter = Counter()
    reasoning_sum_counter = Counter()
    total_sum_counter = Counter()
    for persona in persona_dirs:
        folder = os.path.join(args.input_root, persona)
        files = [f for f in os.listdir(folder) if f.startswith('gsm8k_') and f.endswith('.json')]
        if len(files) != 1319:
            print(f"WARNING: Persona {persona} has {len(files)} JSON files (expected 1319)")
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding='utf-8') as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"SKIP: {persona}/{fname} (invalid or empty JSON) -- {e}", file=sys.stderr)
                continue
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
            extracted, extraction_reason = extract_final_answer(content)
            is_correct = (extracted == gt_num) if (extracted is not None and gt_num is not None) else False
            has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', content))
            has_hashes = bool(re.search(r'####\s*-?[$]?[\d,]+', content))
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
            truncated_by_metadata = (finish_reason == 'length')
            truncated_by_content = is_truly_truncated(content, finish_reason, completion_tokens, max_tokens)
            has_final = has_final_marker(content)
            text_tail = content.strip()[-120:] if content else ''
            extraction_issue = (extracted is None)
            # Error category logic (extraction first, new logic)
            if extracted is not None:
                error_category = 'ok'
            elif truncated_by_content:
                error_category = 'truncated_output'
            else:
                error_category = 'no_number_found'
            error_counter[(persona, error_category)] += 1
            persona_counter[persona] += 1
            total_counter[persona] += 1
            if is_correct:
                correct_counter[persona] += 1
            prompt_sum_counter[persona] += to_number(prompt_tokens)
            completion_sum_counter[persona] += to_number(completion_tokens)
            reasoning_sum_counter[persona] += to_number(reasoning_tokens)
            total_sum_counter[persona] += to_number(total_tokens)
            row = [persona, os.path.relpath(fpath, args.input_root), qid, dataset, gt, gt_num, content, extracted, is_correct, has_boxed, has_hashes, boxed_match, hashes_match, extraction_issue, error_category, extraction_reason, finish_reason, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, truncated_by_metadata, truncated_by_content, has_final, text_tail]
            all_rows.append(row)
    # Write detailed result CSV
    out_csv = os.path.join(args.report_dir, 'results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(["persona_folder","file","question_id","dataset","ground_truth","ground_truth_answer","response_content_full","extracted_answer","is_correct","has_boxed","has_hashes","boxed_match","hashes_match","extraction_issue","error_category","extraction_reason","finish_reason","prompt_tokens","completion_tokens","reasoning_tokens","total_tokens","truncated_by_metadata","truncated_by_content","has_final_marker","text_tail"])
        for row in all_rows:
            writer.writerow(row)

    # Write accuracy summary CSV
    acc_csv = os.path.join(args.report_dir, 'accuracy.csv')
    grand_correct = 0
    grand_total = 0
    grand_prompt_sum = 0.0
    grand_completion_sum = 0.0
    grand_reasoning_sum = 0.0
    grand_total_sum = 0.0
    with open(acc_csv, 'w', newline='', encoding='utf-8') as af:
        writer = csv.writer(af, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            'model_name', 'persona_folder', 'correct', 'total', 'accuracy', 'mean', 'variance (sample variance)', 'std',
            'sum_prompt_tokens', 'sum_completion_tokens', 'sum_reasoning_tokens', 'sum_total_tokens',
            'avg_prompt_tokens', 'avg_completion_tokens', 'avg_reasoning_tokens', 'avg_total_tokens'
        ])
        for persona in PERSONA_LIST:
            correct = correct_counter.get(persona, 0)
            total = total_counter.get(persona, 0)
            acc = (correct / total) if total else 0.0
            mean = acc
            sample_variance = (mean * (1.0 - mean) * total / (total - 1)) if total > 1 else 0.0
            std = math.sqrt(sample_variance)
            sum_prompt_tokens = prompt_sum_counter.get(persona, 0.0)
            sum_completion_tokens = completion_sum_counter.get(persona, 0.0)
            sum_reasoning_tokens = reasoning_sum_counter.get(persona, 0.0)
            sum_total_tokens = total_sum_counter.get(persona, 0.0)
            avg_prompt_tokens = (sum_prompt_tokens / total) if total else 0.0
            avg_completion_tokens = (sum_completion_tokens / total) if total else 0.0
            avg_reasoning_tokens = (sum_reasoning_tokens / total) if total else 0.0
            avg_total_tokens = (sum_total_tokens / total) if total else 0.0
            writer.writerow([
                model_name, persona, correct, total,
                f'{acc:.4f}', f'{mean:.4f}', f'{sample_variance:.8f}', f'{std:.8f}',
                f'{sum_prompt_tokens:.0f}', f'{sum_completion_tokens:.0f}', f'{sum_reasoning_tokens:.0f}', f'{sum_total_tokens:.0f}',
                f'{avg_prompt_tokens:.4f}', f'{avg_completion_tokens:.4f}', f'{avg_reasoning_tokens:.4f}', f'{avg_total_tokens:.4f}'
            ])
            grand_correct += correct
            grand_total += total
            grand_prompt_sum += sum_prompt_tokens
            grand_completion_sum += sum_completion_tokens
            grand_reasoning_sum += sum_reasoning_tokens
            grand_total_sum += sum_total_tokens
        grand_acc = (grand_correct / grand_total) if grand_total else 0.0
        grand_mean = grand_acc
        grand_variance = (grand_mean * (1.0 - grand_mean) * grand_total / (grand_total - 1)) if grand_total > 1 else 0.0
        grand_std = math.sqrt(grand_variance)
        grand_avg_prompt = (grand_prompt_sum / grand_total) if grand_total else 0.0
        grand_avg_completion = (grand_completion_sum / grand_total) if grand_total else 0.0
        grand_avg_reasoning = (grand_reasoning_sum / grand_total) if grand_total else 0.0
        grand_avg_total = (grand_total_sum / grand_total) if grand_total else 0.0
        writer.writerow([
            model_name, 'ALL', grand_correct, grand_total,
            f'{grand_acc:.4f}', f'{grand_mean:.4f}', f'{grand_variance:.8f}', f'{grand_std:.8f}',
            f'{grand_prompt_sum:.0f}', f'{grand_completion_sum:.0f}', f'{grand_reasoning_sum:.0f}', f'{grand_total_sum:.0f}',
            f'{grand_avg_prompt:.4f}', f'{grand_avg_completion:.4f}', f'{grand_avg_reasoning:.4f}', f'{grand_avg_total:.4f}'
        ])

    print(f"Wrote {len(all_rows)} rows to {out_csv}")
    print(f"Wrote accuracy summary to {acc_csv}")
    print("Error category breakdown:")
    for persona in PERSONA_LIST:
        print(f"{persona:20s}", end=' ')
        for cat in ['ok','truncated_output','no_number_found','picked_intermediate_step_bug','other_extraction_error']:
            print(f"{cat}: {error_counter.get((persona,cat),0):4d}", end=' | ')
        print()

if __name__ == '__main__':
    main()
