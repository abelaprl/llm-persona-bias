"""Deep systematic scan of ALL none rows — full text, all positions."""
import sys, re, json, random
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

CSV  = Path("result/benchmarks/ethics/runs/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
ROOT = Path("result/benchmarks/ethics/runs/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
df   = pd.read_csv(CSV)
none_df = df[df["parse_source"] == "none"].reset_index(drop=True)
print(f"Total none: {len(none_df)}\n")

def load_raw(row):
    jf = ROOT / row["persona"] / f"{row['id']}.json"
    if not jf.exists(): return ""
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    return str(d.get("prediction", {}).get("raw", "") or "").strip()

print("Loading full texts...")
data = []
for _, row in none_df.iterrows():
    raw = load_raw(row)
    data.append((row, raw))
print(f"Loaded {len(data)}\n")

NEG = frozenset(["wrong","incorrect","inappropriate","unacceptable"])
POS = frozenset(["right","correct","appropriate","acceptable"])
def wmap(w):
    w = w.lower()
    if w in NEG: return "0"
    if w in POS: return "1"
    return None

out_lines = []
def section(title):
    out_lines.append("\n" + "="*70)
    out_lines.append(title)
    out_lines.append("="*70 + "\n")

# ── 1. Empty / whitespace-only responses ─────────────────────────────────────
section("1. EMPTY / NEAR-EMPTY RESPONSES")
empty = [(row, raw) for row, raw in data if len(raw) < 10]
out_lines.append(f"Count: {len(empty)}")
for row, raw in empty[:5]:
    out_lines.append(f"  [{row['persona']}] GT={row['ground_truth_label']} raw={repr(raw)}")

# ── 2. Texts where digit 0/1 appears ONLY AT VERY START ──────────────────────
section("2. DIGIT ONLY AT START (first 5 chars, not covered by leading_digit)")
pat_start = re.compile(r"^([01])[^0-9A-Za-z]")
start_hits = []
for row, raw in data:
    if not raw: continue
    m = pat_start.match(raw)
    if m and not re.compile(r"^([01])[\s.\)]+[A-Z]").match(raw):  # not already leading_digit
        start_hits.append((row, raw, m.group(1)))
out_lines.append(f"Count: {len(start_hits)}")
for row, raw, pred in start_hits[:10]:
    gt = str(row['ground_truth_label'])
    ok = "OK" if pred == gt else "SALAH"
    out_lines.append(f"  [{ok}] GT={gt} PRED={pred} | {row['persona']}")
    out_lines.append(f"  {repr(raw[:120])}")

# ── 3. Digit alone on its own line (anywhere in text) ────────────────────────
section("3. DIGIT ALONE ON ITS OWN LINE (full text)")
pat_own_line = re.compile(r"(?m)^[ \t]*([01])[ \t]*$")
own_line_hits = []
for row, raw in data:
    if not raw: continue
    ms = list(pat_own_line.finditer(raw))
    if ms:
        pred = ms[-1].group(1)
        gt = str(row['ground_truth_label'])
        own_line_hits.append((row, raw, pred, gt, ms[-1].start()))
out_lines.append(f"Count: {len(own_line_hits)}")
correct = sum(1 for *_, p, g, __ in [(r,raw,p,g,s) for r,raw,p,g,s in own_line_hits] if p==g)
if own_line_hits:
    out_lines.append(f"Accuracy: {correct}/{len(own_line_hits)} = {100*correct/len(own_line_hits):.1f}%")
for row, raw, pred, gt, pos in own_line_hits[:10]:
    ok = "OK" if pred == gt else "SALAH"
    ctx_start = max(0, pos-100)
    ctx = raw[ctx_start:pos+50]
    out_lines.append(f"  [{ok}] GT={gt} PRED={pred} | {row['persona']} | {row['id']}")
    out_lines.append(f"  ...{repr(ctx)}")

# ── 4. "Answer" keyword present but gap too long for current patterns ─────────
section("4. 'ANSWER' PRESENT BUT NOT EXTRACTED (gap analysis)")
pat_ans = re.compile(r"(?i)\banswer\b")
pat_digit_after = re.compile(r"(?i)\banswer\b(.{0,200}?)([01])(?!\d)", re.DOTALL)
ans_with_digit = []
for row, raw in data:
    if not raw: continue
    m = pat_digit_after.search(raw)
    if m:
        gap = len(m.group(1))
        pred = m.group(2)
        gt = str(row['ground_truth_label'])
        ans_with_digit.append((row, raw, pred, gt, gap, m.group(0)[:120]))
out_lines.append(f"Texts with 'answer' + digit within 200 chars: {len(ans_with_digit)}")
# group by gap size
by_gap = defaultdict(list)
for item in ans_with_digit:
    gap = item[4]
    bucket = (gap//20)*20
    by_gap[bucket].append(item)
for bucket in sorted(by_gap):
    items = by_gap[bucket]
    correct_n = sum(1 for r,raw,p,g,gap,ctx in items if p==g)
    out_lines.append(f"  gap {bucket:3d}-{bucket+19:3d} chars: {len(items):4d} items, acc={100*correct_n/len(items):.1f}%")

out_lines.append("\nSamples where gap is 60-120 (just beyond current limit):")
mid_gap = [(r,raw,p,g,gap,ctx) for r,raw,p,g,gap,ctx in ans_with_digit if 60<=gap<=120]
for row, raw, pred, gt, gap, ctx in mid_gap[:8]:
    ok = "OK" if pred == gt else "SALAH"
    out_lines.append(f"  [{ok}] GT={gt} PRED={pred} gap={gap} | {row['persona']}")
    out_lines.append(f"  match: {repr(ctx)}")

# ── 5. "not right/not wrong" double-negation patterns ────────────────────────
section("5. DOUBLE NEGATION: 'not wrong'→1, 'not right'→0")
pat_not_wrong = re.compile(r"(?i)\bnot\s+(?:inherently\s+)?wrong\b")
pat_not_right  = re.compile(r"(?i)\bnot\s+(?:inherently\s+)?right\b")
double_neg = []
for row, raw in data:
    if not raw: continue
    has_not_wrong = bool(pat_not_wrong.search(raw))
    has_not_right  = bool(pat_not_right.search(raw))
    if (has_not_wrong or has_not_right) and not (has_not_wrong and has_not_right):
        pred = "1" if has_not_wrong else "0"
        gt = str(row['ground_truth_label'])
        double_neg.append((row, raw, pred, gt, has_not_wrong))
out_lines.append(f"Count (only one of not_wrong/not_right): {len(double_neg)}")
if double_neg:
    correct = sum(1 for *_, p, g, __ in [(r,raw,p,g,b) for r,raw,p,g,b in double_neg] if p==g)
    out_lines.append(f"Accuracy: {correct}/{len(double_neg)} = {100*correct/len(double_neg):.1f}%")
    for row, raw, pred, gt, has_nw in double_neg[:8]:
        ok = "OK" if pred == gt else "SALAH"
        tail = raw[-200:] if len(raw)>200 else raw
        out_lines.append(f"  [{ok}] GT={gt} PRED={pred} | {row['persona']}")
        out_lines.append(f"  ...{repr(tail[-150:])}")

# ── 6. Texts ending with verdict WORD (no digit) ─────────────────────────────
section("6. TEXTS ENDING WITH VERDICT WORD (last 30 chars)")
pat_end_word = re.compile(
    r"(?i)(?:wrong|incorrect|inappropriate|unacceptable|right|correct|appropriate|acceptable)\s*[.!]?\s*$"
)
end_word_hits = []
for row, raw in data:
    if not raw: continue
    m = pat_end_word.search(raw)
    if m:
        word = m.group(0).strip().rstrip(".!").lower()
        pred = wmap(word)
        if pred:
            gt = str(row['ground_truth_label'])
            end_word_hits.append((row, raw, pred, gt, m.group(0).strip()))
out_lines.append(f"Count: {len(end_word_hits)}")
if end_word_hits:
    correct = sum(1 for r,raw,p,g,w in end_word_hits if p==g)
    out_lines.append(f"Accuracy: {correct}/{len(end_word_hits)} = {100*correct/len(end_word_hits):.1f}%")
    for row, raw, pred, gt, word in end_word_hits[:12]:
        ok = "OK" if pred == gt else "SALAH"
        tail = raw[-200:] if len(raw)>200 else raw
        out_lines.append(f"  [{ok}] GT={gt} PRED={pred} word={repr(word)} | {row['persona']}")
        out_lines.append(f"  ...{repr(tail[-120:])}")

# ── 7. Specific phrases not yet covered ──────────────────────────────────────
section("7. SPECIFIC UNCOVERED PHRASES (full text)")
specific_pats = [
    (re.compile(r"(?i)\bmy\s+(?:final\s+)?(?:answer|response|verdict)\s+is\s+([01])(?!\d)"), "my_answer_is"),
    (re.compile(r"(?i)\bthe\s+(?:final\s+)?(?:answer|verdict|response)\s+(?:here\s+)?is\s+([01])(?!\d)"), "the_answer_is"),
    (re.compile(r"(?i)\bso\s+(?:the\s+)?(?:answer|verdict)\s+(?:would\s+be\s+)?(?:is\s+)?([01])(?!\d)"), "so_answer"),
    (re.compile(r"(?i)\bthis\s+(?:is|would\s+be)\s+(?:a\s+)?([01])(?!\d)"), "this_is_digit"),
    (re.compile(r"(?i)\bmy\s+(?:answer|response)\s+(?:would\s+be\s+|to\s+this\s+is\s+)?([01])(?!\d)"), "my_response"),
    (re.compile(r"(?i)(?:provide|giving|give)\s+(?:a\s+response\s+of\s+|the\s+answer\s+as\s+|an?\s+answer\s+of\s+)([01])(?!\d)"), "provide_answer_of"),
    (re.compile(r"(?i)\bI\s+(?:will\s+)?(?:select|choose|pick|go\s+with)\s+([01])(?!\d)"), "i_select"),
    (re.compile(r"(?i)(?:^|\n)([01])\s*[-–—:]\s*\w"), "digit_dash_word"),
    (re.compile(r"(?i)\bscore\s+(?:of\s+|this\s+as\s+)?([01])(?!\d)"), "score_of"),
    (re.compile(r"(?i)\blabel\s+(?:this\s+)?(?:as\s+|:)?\s*([01])(?!\d)"), "label_as"),
    (re.compile(r"(?i)\b(?:classify|categorize)\s+(?:this\s+)?(?:as\s+)?([01])(?!\d)"), "classify_as"),
    (re.compile(r"(?i)\bthe\s+(?:correct\s+)?(?:binary\s+)?(?:answer|response)\s+(?:is\s+|would\s+be\s+)?([01])(?!\d)"), "the_binary_answer"),
    (re.compile(r"(?i)(?:^|\n)\s*\*\s+([01])\b"), "bullet_digit"),
    (re.compile(r"(?i)\bnot\s+(?:morally\s+|ethically\s+)?(?:justifiable|permissible)\b"), "not_justifiable_lit0"),
    (re.compile(r"(?i)\bmorally\s+(?:justifiable|permissible|acceptable)\b"), "morally_permissible_lit1"),
]

for pat, name in specific_pats:
    hits = []
    for row, raw in data:
        if not raw: continue
        ms = list(pat.finditer(raw))
        if not ms: continue
        m = ms[-1]
        if name.endswith("_lit0"):
            pred = "0"
        elif name.endswith("_lit1"):
            pred = "1"
        else:
            pred = m.group(1)
        gt = str(row['ground_truth_label'])
        hits.append((row, raw, pred, gt, m.group(0).strip()))
    n = len(hits)
    if n == 0: continue
    correct = sum(1 for r,raw,p,g,w in hits if p==g)
    acc = 100*correct/n
    star = " ★" if n>=3 and acc>=70 else ""
    out_lines.append(f"\n[{name}] n={n} acc={acc:.1f}%{star}")
    for row, raw, pred, gt, word in hits[:5]:
        ok = "OK" if pred == gt else "SALAH"
        tail = raw[-150:] if len(raw)>150 else raw
        out_lines.append(f"  [{ok}] GT={gt} PRED={pred} match={repr(word[:60])}")
        out_lines.append(f"  ...{repr(tail[-100:])}")

# ── 8. Persona-level breakdown of none ───────────────────────────────────────
section("8. NONE COUNT BY PERSONA (top 15)")
persona_counts = Counter(row['persona'] for row, _ in data)
for p, c in persona_counts.most_common(15):
    out_lines.append(f"  {p:<35} {c:>4}")

# ── 9. ETHICS subset breakdown ────────────────────────────────────────────────
section("9. NONE COUNT BY ETHICS SUBSET")
subset_counts = Counter(row['subset'] for row, _ in data)
for s, c in subset_counts.most_common():
    out_lines.append(f"  {s:<30} {c:>4}")

# ── 10. Sample texts with ANY extractable-looking structure ──────────────────
section("10. TEXTS WITH STRUCTURED OUTPUT MARKERS NOT YET CAUGHT")
pat_structured = re.compile(
    r"(?i)(?:\*\*[A-Za-z\s]{2,30}\*\*\s*[:\-]|^[A-Z][a-z]+\s*:\s*[01]$|"
    r"answer\s*=\s*[01]|result\s*[:\-]\s*[01]|output\s*[:\-]\s*[01])",
    re.MULTILINE
)
struct_hits = []
for row, raw in data:
    if not raw: continue
    ms = list(pat_structured.finditer(raw))
    if ms:
        struct_hits.append((row, raw, ms[-1].group(0).strip()))
out_lines.append(f"Count: {len(struct_hits)}")
for row, raw, match in struct_hits[:15]:
    tail = raw[-200:] if len(raw)>200 else raw
    out_lines.append(f"  GT={row['ground_truth_label']} match={repr(match[:60])} | {row['persona']}")
    out_lines.append(f"  ...{repr(tail[-120:])}")

Path("deep_scan_none.txt").write_text("\n".join(out_lines), encoding="utf-8")
print(f"Written deep_scan_none.txt ({len(out_lines)} lines)")

# Quick terminal summary
print("\n=== QUICK SUMMARY ===")
print(f"Empty responses: {len(empty)}")
print(f"Digit-at-start (not leading_digit): {len(start_hits)}")
print(f"Digit on own line: {len(own_line_hits)}")
print(f"Answer+digit (gap 60-120): {len(mid_gap)}")
print(f"Double-negation: {len(double_neg)}")
print(f"Ending with verdict word: {len(end_word_hits)}")
