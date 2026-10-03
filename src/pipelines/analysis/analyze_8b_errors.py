"""Analyze extraction errors for meta-llama_llama-3.1-8b-instruct results."""
import pandas as pd
import re
import json
from collections import defaultdict

CSV_PATH = r'result/archive/reports_mixed_legacy\meta-llama_llama-3.1-8b-instruct\results.csv'
OUT_WRONG = r'result/archive/reports_mixed_legacy\meta-llama_llama-3.1-8b-instruct\wrong_answers_only_8b.csv'
OUT_POSSIBLE = r'result/archive/reports_mixed_legacy\meta-llama_llama-3.1-8b-instruct\possible_extraction_errors_8b.csv'

df = pd.read_csv(CSV_PATH)
wrong = df[df['is_correct'] == False].copy()
print(f'Total rows: {len(df)}')
print(f'Total wrong: {len(wrong)}  ({len(wrong)/len(df)*100:.2f}%)')
print(f'Total correct: {len(df)-len(wrong)}  ({(len(df)-len(wrong))/len(df)*100:.2f}%)')
print()

# ------------------------------------------------------------------
# Identify potential extraction errors: gold number appears in tail
# ------------------------------------------------------------------
def gold_in_tail(row):
    tail = str(row.get('text_tail', '') or '')
    gold = str(row.get('ground_truth_answer', '') or '')
    if not gold or not tail or gold.strip() == 'nan':
        return False
    try:
        g = float(gold)
        nums = re.findall(r'-?\d+(?:\.\d+)?', tail)
        return any(abs(float(n) - g) < 0.01 for n in nums)
    except Exception:
        return False

wrong['gold_in_tail'] = wrong.apply(gold_in_tail, axis=1)
possible_ee = wrong[wrong['gold_in_tail']].copy()
genuine_wrong = wrong[~wrong['gold_in_tail']].copy()

print(f'Possible extraction errors (gold in tail, wrong extracted): {len(possible_ee)}')
print(f'Genuine model errors (gold NOT in tail):                    {len(genuine_wrong)}')
print()

# ------------------------------------------------------------------
# Save CSVs
# ------------------------------------------------------------------
wrong.to_csv(OUT_WRONG, index=False)
print(f'Wrote {OUT_WRONG}  ({len(wrong)} rows)')

possible_ee.to_csv(OUT_POSSIBLE, index=False)
print(f'Wrote {OUT_POSSIBLE}  ({len(possible_ee)} rows)')
print()

# ------------------------------------------------------------------
# Error breakdown
# ------------------------------------------------------------------
print('=== error_category distribution (all wrong) ===')
print(wrong['error_category'].value_counts().to_string())
print()

print('=== extraction_reason distribution (all wrong) ===')
print(wrong['extraction_reason'].value_counts().to_string())
print()

print('=== extraction_reason for POSSIBLE EXTRACTION ERRORS ===')
print(possible_ee['extraction_reason'].value_counts().to_string())
print()

print('=== extraction_reason for GENUINE MODEL ERRORS ===')
print(genuine_wrong['extraction_reason'].value_counts().to_string())
print()

# ------------------------------------------------------------------
# Pattern analysis on possible extraction errors
# ------------------------------------------------------------------
print('=== Pattern analysis on possible extraction errors ===')

def classify_pattern(row):
    tail = str(row.get('text_tail', '') or '')
    extracted = str(row.get('extracted_answer', '') or '')
    gold = str(row.get('ground_truth_answer', '') or '')
    reason = str(row.get('extraction_reason', '') or '')

    tail_lower = tail.lower()

    # Leading number (first number in sentence)
    if re.match(r'^\s*\d', tail_lower):
        return 'leading_number'

    # Equation in sentence
    if re.search(r'\d\s*[+\-×x*/]\s*\d|\d\s*=\s*\d', tail):
        return 'eq_in_line'

    # "will take / needs to" verb construction
    if re.search(r'\bwill\s+take\b|\bneeds?\s+to\b|\bit\s+takes?\b', tail_lower):
        return 'need_take'

    # "for N days/weeks" trailing context
    if re.search(r'\bfor\s+\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b', tail_lower):
        return 'trailing_context'

    # "and" parallel sum pattern
    if re.search(r'\d+\s+\w+\s+and\s+\d+\s+\w+', tail_lower):
        return 'parallel_and'

    # "total" keyword present
    if re.search(r'\btotal\b|\boverall\b|\bcombined\b', tail_lower):
        return 'total_keyword'

    # Currency before answer
    if re.search(r'[\$€£]\s*\d', tail):
        return 'currency_present'

    # Relative quantity ("more than", "less than", "fewer")
    if re.search(r'\bmore\s+than\b|\bless\s+than\b|\bfewer\b|\bextra\b', tail_lower):
        return 'relative_quantity'

    return 'unknown'

possible_ee['pattern'] = possible_ee.apply(classify_pattern, axis=1)
pattern_counts = possible_ee['pattern'].value_counts()
print(pattern_counts.to_string())
print()

# ------------------------------------------------------------------
# Sample tails for each pattern
# ------------------------------------------------------------------
print('=== Sample tails for each error pattern ===')
for pat in pattern_counts.index:
    sample = possible_ee[possible_ee['pattern'] == pat].head(3)
    print(f'\n--- {pat} ({pattern_counts[pat]} cases) ---')
    for _, row in sample.iterrows():
        print(f"  qid={row['question_id']} persona={row['persona_folder']}")
        print(f"  gold={row['ground_truth_answer']}  extracted={row['extracted_answer']}  reason={row['extraction_reason']}")
        tail_preview = str(row['text_tail'] or '')[-200:]
        print(f"  tail: {tail_preview!r}")
        print()

# ------------------------------------------------------------------
# Per-persona breakdown of possible extraction errors
# ------------------------------------------------------------------
print('=== Possible extraction errors per persona ===')
persona_ee = possible_ee.groupby('persona_folder').size().reset_index(name='count')
persona_ee = persona_ee.sort_values('count', ascending=False)
print(persona_ee.to_string(index=False))
