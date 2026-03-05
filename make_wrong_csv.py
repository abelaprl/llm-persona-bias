import pandas as pd
import re

df = pd.read_csv('result/reports/meta-llama_llama-3.3-70b-instruct/results.csv')

# Extract qid from file column e.g. "man_explicit_1\gsm8k_00001.json"
df['qid'] = df['file'].str.extract(r'(gsm8k_\d+)', expand=False)

wrong = df[df['is_correct'] == False].copy()

# Key columns
cols = ['persona_folder','qid','ground_truth_answer','extracted_answer',
        'error_category','extraction_reason','text_tail']
wrong_slim = wrong[cols].sort_values(['persona_folder','qid'])
wrong_slim.to_csv('wrong_answers_only.csv', index=False)
print('Saved wrong_answers_only.csv: %d rows' % len(wrong_slim))
print()

# === NO_NUMBER_FOUND ===
no_ext = wrong[wrong['error_category']=='no_number_found'][cols]
print('=== no_number_found (%d cases) ===' % len(no_ext))
for _, row in no_ext.iterrows():
    print('  %s/%s: gold=%s | tail=%s' % (
        row['persona_folder'], row['qid'],
        row['ground_truth_answer'],
        str(row['text_tail'])[:160]
    ))

print()
# === LIKELY EXTRACTION ERRORS ===
# heuristic: gold number appears in tail text but extraction picked something else
fm = wrong[wrong['extraction_reason'].isin(['final_marker','bold_final','tail_sentence','currency_final','latex_result'])].copy()

def looks_like_extraction_error(row):
    tail = str(row['text_tail'])
    gold = str(int(row['ground_truth_answer'])) if str(row['ground_truth_answer']).endswith('.0') else str(row['ground_truth_answer'])
    return bool(re.search(r'\b' + re.escape(gold) + r'\b', tail))

fm['possible_extract_err'] = fm.apply(looks_like_extraction_error, axis=1)
possible_err = fm[fm['possible_extract_err']].sort_values(['persona_folder','qid'])
print('=== Possible extraction errors (gold in tail but extracted wrong): %d ===' % len(possible_err))
for _, row in possible_err.iterrows():
    print('  %s/%s: gold=%s extracted=%s reason=%s | tail=%s' % (
        row['persona_folder'], row['qid'],
        row['ground_truth_answer'],
        row['extracted_answer'],
        row['extraction_reason'],
        str(row['text_tail'])[:160]
    ))

# Save possible extraction errors to separate CSV
possible_err[cols].to_csv('possible_extraction_errors.csv', index=False)
print()
print('Saved possible_extraction_errors.csv: %d rows' % len(possible_err))
