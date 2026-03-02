import os
import re
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# =======================
# Konstanta & Fallback Mapping
# =======================
DEFAULT_MODEL_DIR = r'K:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think'
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_K = 50

GT_COLS = ['ground_truth_answer', 'ground_truth', 'answer', 'gt', 'target']
RESPONSE_COLS = ['response_content', 'response', 'completion', 'model_output', 'output']
EXTRACTED_COLS = ['extracted_answer', 'extracted', 'predicted_answer', 'prediction']
IS_CORRECT_COLS = ['is_correct', 'correct', 'is_match']
QID_COLS = ['question_id', 'id', 'qid', 'question']
COMPLETION_TOKENS_COLS = ['completion_tokens', 'tokens', 'output_tokens']
MAX_TOKENS_COLS = ['max_completion_tokens', 'max_tokens', 'config_max_tokens']

# =======================
# Utilitas Mapping Kolom
# =======================
def find_col(cols, candidates):
    for c in candidates:
        for col in cols:
            if col.lower() == c.lower():
                return col
    for c in candidates:
        for col in cols:
            if c.lower() in col.lower():
                return col
    return None

def safe_float(x):
    try:
        if pd.isnull(x):
            return np.nan
        return float(str(x).replace(',', '').replace(' ', ''))
    except Exception:
        return np.nan

def round_compare(a, b):
    try:
        if np.isnan(a) or np.isnan(b):
            return False
        if abs(a - int(a)) < 1e-6 and abs(b - int(b)) < 1e-6:
            return int(round(a)) == int(round(b))
        return abs(a - b) < 1e-6
    except Exception:
        return False

# =======================
# Ekstraktor Final Answer
# =======================
def extract_final_answer(text):
    if not isinstance(text, str):
        return np.nan
    # 1. \boxed{...}
    boxed = list(re.finditer(r'\\boxed\{([^}]*)', text))
    if boxed:
        last_boxed = boxed[-1].group(1)
        num = extract_last_number(last_boxed)
        if num is not None:
            return num
    # 2. #### <answer>
    hashes = list(re.finditer(r'####\s*([^\n]+)', text))
    if hashes:
        last_hash = hashes[-1].group(1)
        num = extract_last_number(last_hash)
        if num is not None:
            return num
    # 3. Baris dengan Answer: atau Final:
    answer_lines = []
    for line in text.splitlines():
        if re.search(r'(answer|final)\s*[:：]', line, re.I):
            answer_lines.append(line)
    if answer_lines:
        num = extract_last_number(answer_lines[-1])
        if num is not None:
            return num
    # 4. Fallback: angka terakhir di seluruh teks
    num = extract_last_number(text)
    if num is not None:
        return num
    return np.nan

def extract_last_number(s):
    # Normalisasi LaTeX ribuan: {,} dan \, jadi ,
    s = s.replace('{,}', ',').replace('\\,', ',')
    # Hilangkan simbol mata uang
    s = re.sub(r'[\$]', '', s)
    # Ambil semua angka (termasuk ribuan, desimal, negatif)
    matches = list(re.finditer(r'[-+]?\d[\d,]*\.?\d*', s))
    if not matches:
        return None
    last = matches[-1].group(0)
    # Bersihkan koma ribuan
    last = last.replace(',', '')
    # Hilangkan trailing titik
    last = last.rstrip('.')
    try:
        return float(last)
    except Exception:
        return None

# =======================
# Deteksi Truncation
# =======================
def detect_truncation(row, max_tokens):
    # 1. Berdasarkan completion_tokens
    comp_tokens = row.get('completion_tokens', None)
    if comp_tokens is not None:
        try:
            if int(comp_tokens) >= max_tokens:
                return True
        except Exception:
            pass
    # 2. Heuristik teks
    text = str(row.get('response_content', ''))
    if text.count('[') > text.count(']'):
        return True
    if text.count('(') > text.count(')'):
        return True
    if text.endswith(('\\', '$', '{', '(', '=', ':', ',')):
        return True
    if text.rstrip().endswith(('...', '..')):
        return True
    if '\\boxed{' in text and text.count('{') > text.count('}'):
        return True
    return False

# =======================
# Audit Error Category
# =======================
def audit_error_category(row, old_extracted, new_extracted):
    text = str(row.get('response_content', ''))
    thousands_pattern = r'\d{1,3}(,\d{3})+'
    if re.search(thousands_pattern, text):
        try:
            old = safe_float(old_extracted)
            new = safe_float(new_extracted)
            if not np.isnan(old) and not np.isnan(new) and old < new/10:
                return 'thousands_comma_bug'
        except Exception:
            pass
    # picked_first_number_bug
    numbers = list(re.finditer(r'[-+]?\d[\d,]*\.?\d*', text.replace('{,}', ',').replace('\\,', ',')))
    if numbers:
        first = numbers[0].group(0).replace(',', '')
        try:
            old = str(old_extracted)
            if old == first:
                return 'picked_first_number_bug'
        except Exception:
            pass
    # picked_intermediate_step_bug
    if old_extracted and str(old_extracted) in text and '\\boxed{' in text:
        return 'picked_intermediate_step_bug'
    # no_number_found
    if pd.isnull(new_extracted) or new_extracted == '' or str(new_extracted).lower() == 'nan':
        return 'no_number_found'
    return 'other'

# =======================
# Analisis Format Jawaban
# =======================
def analyze_format(df, response_col):
    stats = {}
    responses = df[response_col].astype(str).fillna('')
    n = len(responses)
    stats['count'] = n
    stats['boxed'] = responses.str.contains(r'\\boxed\{').sum() / n if n else 0
    stats['answer_or_final'] = responses.str.contains(r'(answer|final)\s*[:：]', case=False, regex=True).sum() / n if n else 0
    stats['hash4'] = responses.str.contains(r'####').sum() / n if n else 0
    stats['latex_display'] = responses.str.contains(r'\\\[').sum() / n if n else 0
    stats['latex_inline'] = responses.str.contains(r'\\\(').sum() / n if n else 0
    stats['thousands_comma'] = responses.str.contains(r'\d{1,3}(,\d{3})+').sum() / n if n else 0
    # Separator analysis
    sep_counter = Counter()
    for resp in responses:
        for sep in re.findall(r'[\|\;\:\/]{2,}|[\|\;\:\/]', resp):
            sep_counter[sep] += 1
    top_seps = sep_counter.most_common(10)
    stats['top_separators'] = top_seps
    return stats

# =======================
# Main Pipeline
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    model_dir = args.model_dir
    max_tokens = args.max_tokens
    top_k = args.top_k

    persona_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))]
    results_files = []
    for d in persona_dirs:
        f = os.path.join(d, 'results.csv')
        if os.path.exists(f):
            results_files.append(f)

    all_format_stats = []
    global_responses = []
    global_df = []
    audit_rows = []
    improved_examples = []
    still_wrong_examples = []
    truncation_rows = []

    print(f"Found {len(results_files)} persona results.csv files.")

    for file in results_files:
        persona = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        orig_cols = list(df.columns)
        # Mapping kolom
        qid_col = find_col(orig_cols, QID_COLS)
        gt_col = find_col(orig_cols, GT_COLS)
        response_col = find_col(orig_cols, RESPONSE_COLS)
        extracted_col = find_col(orig_cols, EXTRACTED_COLS)
        is_correct_col = find_col(orig_cols, IS_CORRECT_COLS)
        comp_tokens_col = find_col(orig_cols, COMPLETION_TOKENS_COLS)
        max_tokens_col = find_col(orig_cols, MAX_TOKENS_COLS)

        # Fallback kolom
        if not all([qid_col, gt_col, response_col, extracted_col, is_correct_col]):
            print(f"Warning: {file} missing required columns. Skipped.")
            continue

        # Rename kolom untuk konsistensi
        df = df.rename(columns={
            qid_col: 'question_id',
            gt_col: 'ground_truth_answer',
            response_col: 'response_content',
            extracted_col: 'extracted_answer',
            is_correct_col: 'is_correct'
        })
        if comp_tokens_col:
            df = df.rename(columns={comp_tokens_col: 'completion_tokens'})
        if max_tokens_col:
            df = df.rename(columns={max_tokens_col: 'max_tokens'})

        # Analisis format
        fmt_stats = analyze_format(df, 'response_content')
        fmt_stats['persona'] = persona
        all_format_stats.append(fmt_stats)
        global_responses.extend(df['response_content'].astype(str).tolist())

        # Ekstraksi baru & audit
        new_extracted = []
        new_is_correct = []
        is_truncated = []
        error_category = []
        for idx, row in df.iterrows():
            gt = safe_float(row['ground_truth_answer'])
            resp = row['response_content']
            old_extr = row['extracted_answer']
            new_extr = extract_final_answer(resp)
            new_extracted.append(new_extr)
            # Pembanding
            if np.isnan(gt) or new_extr is None or str(new_extr).lower() == 'nan':
                new_is_correct.append(False)
            else:
                new_is_correct.append(round_compare(new_extr, gt))
            # Truncation
            row_dict = row.to_dict()
            row_dict['completion_tokens'] = row.get('completion_tokens', None)
            row_dict['response_content'] = resp
            is_truncated.append(detect_truncation(row_dict, max_tokens))
            # Audit error
            error_category.append(audit_error_category(row_dict, old_extr, new_extr))

        df['new_extracted_answer'] = new_extracted
        df['new_is_correct'] = new_is_correct
        df['is_truncated'] = is_truncated
        df['error_category'] = error_category

        # Audit & contoh
        old_acc = np.mean(df['is_correct'].astype(str).str.lower() == 'true')
        new_acc = np.mean(df['new_is_correct'])
        old_wrong = df[df['is_correct'].astype(str).str.lower() != 'true']
        new_wrong = df[~df['new_is_correct']]
        improved = df[(df['is_correct'].astype(str).str.lower() != 'true') & (df['new_is_correct'])]
        regressed = df[(df['is_correct'].astype(str).str.lower() == 'true') & (~df['new_is_correct'])]

        # Simpan contoh improved dan still wrong
        improved_sorted = improved.copy()
        improved_sorted['has_thousands'] = improved_sorted['response_content'].str.contains(r'\d{1,3}(,\d{3})+')
        improved_sorted = improved_sorted.sort_values(['has_thousands'], ascending=False)
        improved_examples.extend(improved_sorted.head(top_k).assign(file=file).to_dict('records'))

        still_wrong_sorted = new_wrong.copy()
        still_wrong_sorted = still_wrong_sorted.sort_values(['is_truncated'], ascending=False)
        still_wrong_examples.extend(still_wrong_sorted.head(top_k).assign(file=file).to_dict('records'))

        # Truncation report
        trunc = df[df['is_truncated']]
        for _, r in trunc.iterrows():
            truncation_rows.append({
                'persona': persona,
                'question_id': r['question_id'],
                'is_correct': r['is_correct'],
                'new_is_correct': r['new_is_correct'],
                'is_truncated': r['is_truncated'],
                'completion_tokens': r.get('completion_tokens', None)
            })

        # Audit summary
        for _, r in df.iterrows():
            audit_rows.append({
                'persona': persona,
                'file': file,
                'question_id': r['question_id'],
                'ground_truth_answer': r['ground_truth_answer'],
                'extracted_answer': r['extracted_answer'],
                'new_extracted_answer': r['new_extracted_answer'],
                'is_correct': r['is_correct'],
                'new_is_correct': r['new_is_correct'],
                'is_truncated': r['is_truncated'],
                'error_category': r['error_category']
            })

        # Simpan file fixed
        out_file = os.path.join(os.path.dirname(file), 'results_fixed.csv')
        df.to_csv(out_file, index=False)

        # Gabung global
        global_df.append(df)

        print(f"{persona}: old_acc={old_acc:.3f}, new_acc={new_acc:.3f}, improved={len(improved)}, regressed={len(regressed)}, truncated={df['is_truncated'].sum()}")

    # Output analisis format
    fmt_df = pd.DataFrame(all_format_stats)
    fmt_df.to_csv(os.path.join(model_dir, 'format_stats_by_persona.csv'), index=False)

    # Global format
    global_fmt = analyze_format(pd.DataFrame({'response_content': global_responses}), 'response_content')
    pd.DataFrame([global_fmt]).to_csv(os.path.join(model_dir, 'format_stats_global.csv'), index=False)

    # Output audit summary
    pd.DataFrame(audit_rows).to_csv(os.path.join(model_dir, 'extraction_audit_summary.csv'), index=False)
    pd.DataFrame(improved_examples).to_csv(os.path.join(model_dir, 'examples_improved.csv'), index=False)
    pd.DataFrame(still_wrong_examples).to_csv(os.path.join(model_dir, 'examples_still_wrong.csv'), index=False)
    pd.DataFrame(truncation_rows).to_csv(os.path.join(model_dir, 'truncation_report.csv'), index=False)

    # Rekap global
    if global_df:
        all_df = pd.concat(global_df, ignore_index=True)
        old_acc = np.mean(all_df['is_correct'].astype(str).str.lower() == 'true')
        new_acc = np.mean(all_df['new_is_correct'])
        total_trunc = all_df['is_truncated'].sum()
        print("\n=== Summary ===")
        print(f"Total persona processed: {len(results_files)}")
        print(f"Global old accuracy: {old_acc:.3%}")
        print(f"Global new accuracy: {new_acc:.3%}")
        print(f"Total truncated: {total_trunc} ({total_trunc/len(all_df):.2%})")
        print(f"Output files written to: {model_dir}")

if __name__ == '__main__':
    main()
