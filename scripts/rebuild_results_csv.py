import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default=r'E:\Tugas Akhir\TugasAkhir\result\allenai_olmo-3-32b-think')
    parser.add_argument('--report_dir', default=r'E:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think')
    parser.add_argument('--pattern', default='gsm8k_*.json')
    parser.add_argument('--examples_still_wrong', default=None)
    return parser.parse_args()

def clean_text(text):
    if text is None:
        return ''
    return text.replace('\x00', '')

def parse_ground_truth(gt):
    if not gt:
        return None
    # Try ####
    m = re.search(r'####\s*([-+]?[\d,]+(?:\.\d+)?)', gt)
    if m:
        return float(m.group(1).replace(',', ''))
    # Else last number
    nums = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', gt)
    if nums:
        return float(nums[-1].replace(',', ''))
    return None

def extract_boxed(content):
    # Find all boxed forms
    boxed = re.findall(r'\\boxed\{([^}]*)\}|\\boxed\s*\{([^}]*)\}|\$\\boxed\{([^}]*)\}\$', content)
    for group in boxed[::-1]:
        for val in group:
            if val:
                num = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', val)
                # Filter out empty strings (shouldn't happen, but safety)
                num = [n for n in num if n and n.replace(',', '').replace('.', '').strip()]
                if num:
                    try:
                        return float(num[-1].replace(',', ''))
                    except Exception:
                        continue
    return None

def extract_after_hashes(content):
    m = re.search(r'####\s*([-+]?[\d,]+(?:\.\d+)?)', content)
    if m:
        return float(m.group(1).replace(',', ''))
    return None

def extract_after_answer_label(content):
    for line in content.splitlines():
        if re.match(r'\s*(Answer:|Final answer:)', line, re.I):
            nums = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', line)
            if nums:
                return float(nums[0].replace(',', ''))
    return None

def extract_last_number(content):
    # Remove notes section if present
    notes_kw = ['note', 'notes', 'remark', 'footnote', 'extra', 'side note']
    lower = content.lower()
    for kw in notes_kw:
        idx = lower.rfind(kw)
        if idx != -1:
            content = content[:idx]
    nums = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', content)
    # Filter out empty strings (shouldn't happen, but safety)
    nums = [n for n in nums if n and n.replace(',', '').replace('.', '').strip()]
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except Exception:
            return None
    return None

def robust_extractor(content):
    extraction_reason = ''
    boxed = extract_boxed(content)
    if boxed is not None:
        extraction_reason = 'boxed'
        return boxed, extraction_reason
    hashes = extract_after_hashes(content)
    if hashes is not None:
        extraction_reason = '####'
        return hashes, extraction_reason
    ans = extract_after_answer_label(content)
    if ans is not None:
        extraction_reason = 'answer_label'
        return ans, extraction_reason
    # Fallback: last number, but only if no boxed/####/Answer
    last = extract_last_number(content)
    if last is not None:
        extraction_reason = 'last_number_fallback'
        return last, extraction_reason
    extraction_reason = 'no_number_found'
    return None, extraction_reason

def is_truncated(resp, content, max_tokens, completion_tokens, finish_reason, native_finish_reason):
    if finish_reason == 'length' or native_finish_reason == 'length':
        return True
    if completion_tokens is not None and max_tokens is not None and completion_tokens >= max_tokens - 5:
        return True
    # Heuristic: abrupt ending
    if content:
        last = content.strip()[-50:]
        if not re.search(r'[.!?}\]\)]$', last) and not re.search(r'(boxed|####|Answer:|Final answer:)', last, re.I):
            return True
    return False

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)
    all_rows = []
    still_wrong = []
    trunc_ids = set()
    not_trunc_ids = set()
    error_counter = Counter()
    persona_counter = Counter()
    for root, dirs, files in os.walk(args.input_root):
        persona = os.path.basename(root)
        jsons = [f for f in files if f.endswith('.json')]
        if not jsons:
            continue
        print(f"Persona {persona}: {len(jsons)} JSON files")
        for fname in jsons:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding='utf-8') as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")
                continue
            meta = data.get('metadata', {})
            inp = data.get('input', {})
            req = data.get('request', {})
            resp = data.get('response', {})
            model_id = meta.get('model_info', {}).get('model_id', '')
            model_name = meta.get('model_info', {}).get('model_name', '')
            persona_type = inp.get('persona', {}).get('type', '')
            persona_id = inp.get('persona', {}).get('id', '')
            dataset = inp.get('dataset', '')
            question_id = inp.get('question_id', '')
            ground_truth_raw = inp.get('ground_truth', '')
            ground_truth_answer = parse_ground_truth(ground_truth_raw)
            json_path = os.path.abspath(fpath)
            response_content_full = clean_text(resp.get('content', ''))
            response_success = resp.get('success', False)
            finish_reason = None
            native_finish_reason = None
            try:
                choices = resp.get('raw_api_response', {}).get('choices', [{}])
                if choices and isinstance(choices, list):
                    finish_reason = choices[0].get('finish_reason', None)
                    native_finish_reason = choices[0].get('native_finish_reason', None)
            except Exception:
                pass
            max_tokens = req.get('parameters', {}).get('max_tokens', None)
            completion_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('completion_tokens', None)
            prompt_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('prompt_tokens', None)
            total_tokens = resp.get('raw_api_response', {}).get('usage', {}).get('total_tokens', None)
            extracted_answer, extraction_reason = robust_extractor(response_content_full)
            is_correct = None
            error_category = 'ok'
            # Truncation
            is_trunc = is_truncated(resp, response_content_full, max_tokens, completion_tokens, finish_reason, native_finish_reason)
            if is_trunc:
                trunc_ids.add(str(question_id))
            else:
                not_trunc_ids.add(str(question_id))
            # Extraction error logic
            if extracted_answer is None:
                error_category = 'no_number_found' if response_content_full.strip() else 'other_extraction_error'
            elif is_trunc and (extracted_answer is None or not response_content_full.strip()):
                error_category = 'truncated_output'
            # Correctness
            if extracted_answer is not None and ground_truth_answer is not None:
                if abs(extracted_answer - ground_truth_answer) < 1e-6:
                    is_correct = True
                else:
                    is_correct = False
            # picked_intermediate_step_bug detection
            # (If boxed/####/Answer exists, but last-number fallback would have picked a different number)
            boxed = extract_boxed(response_content_full)
            hashes = extract_after_hashes(response_content_full)
            ans = extract_after_answer_label(response_content_full)
            last = extract_last_number(response_content_full)
            if (boxed is not None or hashes is not None or ans is not None) and last is not None:
                canonical = boxed or hashes or ans
                if last != canonical and extracted_answer == canonical and (is_correct is False):
                    error_category = 'picked_intermediate_step_bug'
            # Final error_category
            if is_trunc and (extracted_answer is None or not response_content_full.strip()):
                error_category = 'truncated_output'
            if error_category == 'ok' and not is_correct:
                error_category = 'other_extraction_error'
            error_counter[error_category] += 1
            persona_counter[persona_type] += 1
            row = [model_id, model_name, persona_type, persona_id, dataset, question_id, json_path, ground_truth_raw, ground_truth_answer, response_success, finish_reason, native_finish_reason, max_tokens, completion_tokens, prompt_tokens, total_tokens, is_trunc, extracted_answer, is_correct, error_category, extraction_reason, len(response_content_full), response_content_full]
            all_rows.append(row)
            if is_correct is False or extracted_answer is None:
                still_wrong.append([json_path, persona_type, question_id, ground_truth_answer, extracted_answer, error_category, extraction_reason, is_trunc, finish_reason, response_content_full])
    # Write results.csv
    out_csv = os.path.join(args.report_dir, 'results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(["model_id","model_name","persona_type","persona_id","dataset","question_id","json_path","ground_truth_raw","ground_truth_answer","response_success","finish_reason","native_finish_reason","max_tokens","completion_tokens","prompt_tokens","total_tokens","is_truncated","extracted_answer","is_correct","error_category","extraction_reason","response_char_len","response_content_full"])
        for row in all_rows:
            writer.writerow(row)
    # Write examples_still_wrong.csv
    out_wrong = os.path.join(args.report_dir, 'examples_still_wrong.csv')
    with open(out_wrong, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(["json_path","persona_type","question_id","ground_truth_answer","extracted_answer","error_category","extraction_reason","is_truncated","finish_reason","response_content_full"])
        for row in still_wrong:
            writer.writerow(row)
    # Write truncated_question_ids.csv
    with open(os.path.join(args.report_dir, 'truncated_question_ids.csv'), 'w', encoding='utf-8') as f:
        for qid in sorted(trunc_ids):
            f.write(str(qid)+'\n')
    with open(os.path.join(args.report_dir, 'not_truncated_question_ids.csv'), 'w', encoding='utf-8') as f:
        for qid in sorted(not_trunc_ids):
            f.write(str(qid)+'\n')
    # Write summary.json
    summary = {
        'total': len(all_rows),
        'per_persona': dict(persona_counter),
        'error_category': dict(error_counter),
        'truncation_rate': len(trunc_ids) / max(1, len(all_rows)),
    }
    with open(os.path.join(args.report_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(all_rows)} rows to {out_csv}")
    print(f"Truncation rate: {summary['truncation_rate']:.3%}")
    print(f"Top error categories: {error_counter.most_common(5)}")

if __name__ == '__main__':
    main()
