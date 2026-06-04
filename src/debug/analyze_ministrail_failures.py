"""
Analyze no_number_found failures in ministrail rebuild CSVs.
Classifies each failure and produces a breakdown table.
"""
import csv, re, sys, collections, os

CSV_FILES = [
    r'result/reports/GSM8K/_tmp_rebuild_results_csv_ministrail/results.csv',
]

failed_rows = []
all_rows = []
for csv_path in CSV_FILES:
    try:
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                if row.get('error_category', '').strip() == 'no_number_found':
                    row['_source'] = csv_path
                    failed_rows.append(row)
    except FileNotFoundError:
        print(f"MISSING: {csv_path}")

print(f"Total rows scanned: {len(all_rows)}")
print(f"no_number_found failures: {len(failed_rows)}")
print()

WORD_NUMS = (
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
    r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
    r'eighty|ninety|hundred|thousand|million)\b'
)

REFUSAL_PATS = [
    r'this problem (cannot|can\'t|is impossible|does not have|has no valid)',
    r'there is (no|not enough) (valid |unique |clear |)?(?:answer|solution|information)',
    r'impossible to (solve|determine|calculate)',
    r'the problem (as stated|statement) (is|seems|appears|does not|has)',
    r'is likely (a )?typo|there may be (a )?typo|there is (a )?typo',
    r'cannot be solved|no solution|no valid answer',
    r'need more information|not enough information|missing information',
    r'insufficient (data|information)',
    r'no (unique|valid|specific|definitive) (answer|solution)',
]

MARKER_PATS = [
    r'(?:Final\s+)?Answer\s*:',
    r'Therefore\b',
    r'Thus\b',
    r'Hence\b',
    r'So the answer',
]

NUMBER_RE = r'(?:[-]?\s*)?(?:\$)?(?:\d{1,3}(?:(?:,|\s)\d{3})+|\d+)(?:\.\d+)?'


def has_marker(text):
    return any(re.search(p, text, re.I) for p in MARKER_PATS)


def classify_failure(row):
    content = row.get('response_content_full', '') or ''
    gt_str = row.get('ground_truth_answer', '') or ''

    if not content or not content.strip():
        return 'A_empty_response'

    lines = content.strip().splitlines()
    last20 = lines[-20:]
    last5 = lines[-5:]
    tail = content.strip()[-600:]

    last_nonempty = ''
    for ln in reversed(lines):
        if ln.strip():
            last_nonempty = ln.strip()
            break

    # ── A: Empty / API artifact ──────────────────────────────────────────────
    if len(content.strip()) < 10:
        return 'A_empty_response'

    # ── B: Model refusal / caveat / "no valid solution" ──────────────────────
    for pat in REFUSAL_PATS:
        if re.search(pat, content, re.I):
            return 'B_model_refusal_or_caveat'

    # ── C: LaTeX fraction / complex math result that extractor can't parse ───
    # \frac, \sqrt, \left, \right — these patterns produce e.g. \frac{5}{3}
    # as the final value which is not a plain integer
    if re.search(r'\\frac\{|\\sqrt\{', tail):
        return 'C_latex_fraction_result'

    # ── D: Plain number IS the last non-empty line but rule 5d missed it ─────
    # Rule 5d requires match of m.group(1) but some edge cases slip through
    if re.match(r'^\d+\.?\d*\s*$', last_nonempty):
        return 'D_plain_number_last_line_missed'

    # ── E: Number at end of last line, rule 5 tail_sentence missed ───────────
    # e.g. "...so Joe has 24 apples."  or  "...C = 94."
    has_num_in_tail = bool(re.search(r'\d', tail))
    has_marker_in_tail = has_marker(tail)

    # Check last line ends with number pattern
    if re.search(r'=\s*\d+\.?\d*\s*[.!]?\s*$', last_nonempty):
        return 'E_equals_number_at_eol'

    if re.search(r'\d+\.?\d*\s*[.!]?\s*$', last_nonempty) and has_marker_in_tail:
        return 'F_marker_present_number_at_eol'

    # ── F: Marker in tail but extraction still failed ────────────────────────
    if has_marker_in_tail and has_num_in_tail:
        return 'F_marker_present_extraction_failed'

    # ── G: Answer appears too early (not in last 20 lines) ───────────────────
    if len(lines) > 25:
        early_text = '\n'.join(lines[:-20])
        if has_marker(early_text) and re.search(r'\d', early_text):
            # Check if tail has NO marker
            if not has_marker_in_tail:
                return 'G_answer_too_early_no_tail_marker'

    # ── H: Word number only in last 5 lines ──────────────────────────────────
    last5_text = '\n'.join(last5)
    if re.search(WORD_NUMS, last5_text, re.I) and not re.search(r'\d', last5_text):
        return 'H_word_number_only'

    # ── I: Number in tail, no marker, extractor rule 5 missed prose ──────────
    if has_num_in_tail and not has_marker_in_tail:
        # Is there a sentence ending with number?
        if re.search(r'\d+\.?\d*\s*[.!]?\s*$', last_nonempty):
            return 'I_prose_answer_no_marker'
        # Number somewhere in tail but not at end
        return 'I_number_in_tail_not_at_eol'

    # ── J: No number at all in the tail ──────────────────────────────────────
    if not has_num_in_tail:
        return 'J_no_number_in_tail_at_all'

    return 'Z_other'


categories = collections.Counter()
examples = collections.defaultdict(list)

for row in failed_rows:
    cat = classify_failure(row)
    categories[cat] += 1
    content = row.get('response_content_full', '') or ''
    if len(examples[cat]) < 5:
        examples[cat].append({
            'gt': row.get('ground_truth_answer', ''),
            'tail': content.strip()[-350:] if content else '',
            'last_line': (content.strip().splitlines() or [''])[-1][:120],
            'persona': row.get('persona_folder', ''),
        })

CATEGORY_LABELS = {
    'A_empty_response':               'A. Empty/null response',
    'B_model_refusal_or_caveat':      'B. Model refusal / "no valid solution"',
    'C_latex_fraction_result':        'C. LaTeX fraction final value (\\frac)',
    'D_plain_number_last_line_missed':'D. Plain number last line — rule 5d missed',
    'E_equals_number_at_eol':         'E. "= N" at end of last line, no rule caught',
    'F_marker_present_number_at_eol': 'F. Marker present, number at EOL missed',
    'F_marker_present_extraction_failed': 'F. Marker present, extraction still failed',
    'G_answer_too_early_no_tail_marker':'G. Answer only in early text, tail has no marker',
    'H_word_number_only':             'H. Word number only (e.g. "forty-two")',
    'I_prose_answer_no_marker':       'I. Prose answer, no marker, rule 5 missed',
    'I_number_in_tail_not_at_eol':    'I. Number in tail but not at EOL',
    'J_no_number_in_tail_at_all':     'J. No number at all in last 600 chars',
    'Z_other':                        'Z. Other / unclassified',
}

total = len(failed_rows)
print("=" * 80)
print(f"{'Failure Type':<50} {'Count':>6} {'%':>6}")
print("=" * 80)
for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
    label = CATEGORY_LABELS.get(cat, cat)
    print(f"{label:<50} {cnt:>6}  {cnt/total*100:>5.1f}%")
print("-" * 80)
print(f"{'TOTAL':<50} {total:>6}  100.0%")
print()

# Print examples per category
print("=" * 80)
print("EXAMPLES PER CATEGORY")
print("=" * 80)
for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
    label = CATEGORY_LABELS.get(cat, cat)
    print(f"\n[{label}]  n={cnt}")
    for i, ex in enumerate(examples[cat][:3]):
        print(f"  Example {i+1} | GT={ex['gt']} | persona={ex['persona']}")
        print(f"  Last line: {ex['last_line']!r}")
        print(f"  Tail: {ex['tail']!r}")
        print()
