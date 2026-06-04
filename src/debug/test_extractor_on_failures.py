"""
Run the current extract_final_answer on the 42 recoverable failures.
Shows exactly where each one fails so we can design targeted fixes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rebuild'))

# Import the extractor from the ministrail rebuild script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rebuild_ministrail",
    os.path.join(os.path.dirname(__file__), '..', 'rebuild', 'rebuild_results_csv_ministrail.py')
)
mod = importlib.util.load_from_spec = None

# Inline the key functions instead
import csv, re

NUMBER_RE = r'(?:[-−–]\s*)?(?:\$)?(?:\d{1,3}(?:(?:,\\!|,|\{,\}|\s)\d{3})+|\d+)(?:\.\d+)?'

def normalize_number(raw):
    if raw is None:
        return None
    s = str(raw)
    s = s.replace(' ', ' ').replace(' ', ' ').replace(' ', ' ')
    s = re.sub(r'\{,\}', ',', s)
    s = s.replace('\\!', '')
    s = s.replace('$', '').replace('€', '').replace('\xa3', '').replace('%', '').strip()
    s = s.replace('−', '-').replace('–', '-').replace('- ', '-')
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

csv_path = r'result/reports/GSM8K/_tmp_rebuild_results_csv_ministrail/results.csv'

rows = []
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ec = row.get('error_category', '') or ''
        content = row.get('response_content_full', '') or ''
        if ec.strip() == 'no_number_found' and content.strip():
            rows.append(row)

print(f'Testing {len(rows)} recoverable failures\n')

# For each, show: last line, what NUMBER_RE matches, why tail_sentence/marker fails
for i, row in enumerate(rows):
    content = row.get('response_content_full', '') or ''
    gt = row.get('ground_truth_answer', '')
    persona = row.get('persona_folder', '')
    fname = row.get('file', '')

    lines = content.strip().splitlines()
    last_nonempty = ''
    for ln in reversed(lines):
        if ln.strip():
            last_nonempty = ln.strip()
            break

    tail = content.strip()[-600:]
    tail_lines = tail.splitlines()[-5:]
    last_tail_line = tail_lines[-1].strip() if tail_lines else ''

    # What does NUMBER_RE find in last_tail_line?
    nums_found = [(m.group(0), normalize_number(m.group(0))) for m in re.finditer(NUMBER_RE, last_tail_line)]

    # Does last line start with a number?
    starts_num = bool(re.match(r'^\d', last_tail_line))

    # Does last line end with a number?
    ends_num = bool(re.search(r'\d\s*[.!]?\s*$', last_tail_line))

    # Is there a percent pattern?
    has_pct = bool(re.search(r'\d+%', last_tail_line))

    # Is there an ordinal?
    has_ordinal = bool(re.search(r'\d+(st|nd|rd|th)\b', last_tail_line, re.I))

    # Is there a word number?
    WNUMS = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b'
    word_nums = re.findall(WNUMS, last_tail_line, re.I)

    # Classify
    flags = []
    if has_pct: flags.append('PERCENT')
    if has_ordinal: flags.append('ORDINAL')
    if word_nums: flags.append(f'WORDNUM:{word_nums}')
    if not nums_found and not word_nums: flags.append('NO_DIGIT')
    if nums_found and not ends_num: flags.append('NUM_NOT_AT_EOL')
    if ends_num: flags.append('ENDS_WITH_NUM')

    print(f'#{i+1:02d} GT={gt:>4} | {",".join(flags) or "?"}')
    print(f'      last_line: {last_tail_line[:120]!r}')
    if nums_found:
        print(f'      nums_re:   {nums_found}')
    print()
