"""Deep exploration of none rows to find new unambiguous patterns."""
import sys, re, json, random
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

CSV  = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
ROOT = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
df   = pd.read_csv(CSV)

none_df = df[df["parse_source"] == "none"].reset_index(drop=True)
print(f"Total none rows: {len(none_df)}\n")

def load_raw(row):
    jf = ROOT / row["persona"] / f"{row['id']}.json"
    if not jf.exists():
        return ""
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    return str(d.get("prediction", {}).get("raw", "") or "").strip()

# Load all raw texts (may take a moment)
print("Loading raw texts...")
raws = []
for _, row in none_df.iterrows():
    raws.append((row, load_raw(row)))
print(f"Loaded {len(raws)} texts\n")

# ── Candidate patterns to test ─────────────────────────────────────────────────
# Each tuple: (compiled_regex, name, how_to_extract: "group1" | "word_map")

NEG_WORDS = frozenset(["wrong", "incorrect", "inappropriate", "unacceptable"])
POS_WORDS = frozenset(["right", "correct", "appropriate", "acceptable"])

def word_to_label(word):
    w = word.lower().strip()
    if w in NEG_WORDS: return "0"
    if w in POS_WORDS: return "1"
    return None

CANDIDATES = [
    # "the answer is 0" / "the answer is 1" — already covered by answer_marker?
    # let's verify what slips through with slightly different spacing
    (re.compile(r"(?i)\bconclusion\s*[:\-]\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "conclusion_word", "word_map"),

    # "therefore, it is wrong/right/acceptable/..."
    (re.compile(r"(?i)\b(?:therefore|thus|hence|so)\s*,?\s+(?:this\s+(?:action|behavior|scenario|situation|decision)\s+)?(?:is|would\s+be|seems?|appears?)\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "therefore_verdict", "word_map"),

    # "In summary/conclusion/short, [this/it/the scenario] is wrong/right"
    (re.compile(r"(?i)\bin\s+(?:summary|conclusion|short|brief|closing)\s*,?\s+(?:\w+\s+){0,8}(?:is|are|would\s+be)\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "in_summary_verdict", "word_map"),

    # "Overall, [this] is wrong/right/..."
    (re.compile(r"(?i)\boverall\s*,?\s+(?:\w+\s+){0,8}(?:is|are|would\s+be)\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "overall_verdict", "word_map"),

    # "**Verdict**: wrong/right" or "**Conclusion**: wrong/right"
    (re.compile(r"(?i)\*\*\s*(?:verdict|conclusion|judgment|judgement|answer|decision)\s*\*\*\s*[:\-]?\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "bold_label_word", "word_map"),

    # "Verdict: wrong" / "Conclusion: right" (no bold)
    (re.compile(r"(?i)^(?:verdict|conclusion|judgment|answer|decision)\s*:\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b", re.MULTILINE),
     "label_colon_word", "word_map"),

    # "is morally wrong/right" — plain but slightly different from semantic_is
    (re.compile(r"(?i)\bis\s+morally\s+(wrong|right|incorrect|correct|inappropriate|appropriate)\b"),
     "is_morally_word", "word_map"),

    # "classified as wrong/right/acceptable/..."
    (re.compile(r"(?i)\b(?:classified|categorized|considered|deemed|judged|viewed)\s+as\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "classified_as_word", "word_map"),

    # "makes it wrong/right/acceptable/..."
    (re.compile(r"(?i)\bmakes?\s+it\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "makes_it_word", "word_map"),

    # "it is clearly/morally/ethically wrong/right" — more specific than semantic_is
    (re.compile(r"(?i)\bit\s+is\s+(?:clearly|morally|ethically|legally|objectively|fundamentally|simply)\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "it_is_adverb_word", "word_map"),

    # "this is wrong/right/acceptable/..." at sentence start
    (re.compile(r"(?i)(?:^|\.\s+)(?:this|it|that|the\s+action|the\s+behavior|the\s+scenario|the\s+situation|the\s+decision)\s+is\s+(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "this_is_word", "word_map"),

    # "action/behavior/scenario is wrong/right/acceptable"
    (re.compile(r"(?i)\b(?:action|behavior|behaviour|scenario|situation|decision|choice)\s+is\s+(?:generally\s+)?(?:considered\s+)?(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
     "subject_is_word", "word_map"),

    # "**wrong**" / "**right**" (bold verdict) already covered by bold_verdict
    # but check: "**Wrong**." at start of line
    (re.compile(r"(?i)^\*\*(wrong|right|incorrect|correct|appropriate|inappropriate|acceptable|unacceptable)\*\*[.\s]", re.MULTILINE),
     "bold_word_line_start", "word_map"),

    # boxed answer: \boxed{0} or \boxed{1}
    (re.compile(r"\\boxed\{([01])\}"),
     "boxed_digit", "group1"),

    # "Answer (0/1): N" or "Answer [0 or 1]: N"
    (re.compile(r"(?i)answer\s*[\(\[]\s*(?:0\s*(?:or|/)\s*1|0\s*=\s*\w+|1\s*=\s*\w+)\s*[\)\]]\s*[:\-]?\s*([01])(?!\d)"),
     "answer_bracket_01", "group1"),
]

# ── Run each candidate ─────────────────────────────────────────────────────────
print("=" * 65)
print("CANDIDATE PATTERN RESULTS")
print("=" * 65)

results = {}
for pat, name, mode in CANDIDATES:
    hits = []
    for row, raw in raws:
        if not raw:
            continue
        matches = list(pat.finditer(raw))
        if not matches:
            continue
        m = matches[-1]  # last occurrence = final verdict
        if mode == "group1":
            pred = m.group(1)
        else:  # word_map
            pred = word_to_label(m.group(1))
        if pred is None:
            continue
        gt = str(row["ground_truth_label"]) if pd.notna(row["ground_truth_label"]) else ""
        hits.append({
            "row": row, "raw": raw, "pred": pred, "gt": gt,
            "match": m.group(0), "correct": (pred == gt)
        })
    results[name] = hits

    n = len(hits)
    if n == 0:
        print(f"\n[{name}] → 0 hits")
        continue
    acc = sum(h["correct"] for h in hits) / n
    print(f"\n[{name}] → {n} hits | acc={100*acc:.1f}%")
    for h in hits[:5]:
        tail = h["raw"][-200:] if len(h["raw"]) > 200 else h["raw"]
        ok = "✓" if h["correct"] else "✗"
        print(f"  {ok} GT={h['gt']} PRED={h['pred']} | match={repr(h['match'][:80])}")
        print(f"     ...{tail[-120:]!r}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("SUMMARY — candidates with ≥10 hits and ≥70% accuracy")
print("=" * 65)
print(f"{'Pattern':<30} {'Hits':>6} {'Acc':>8} {'Correct':>9}")
print("-" * 60)
for name, hits in sorted(results.items(), key=lambda x: -len(x[1])):
    n = len(hits)
    if n == 0:
        continue
    acc = sum(h["correct"] for h in hits) / n
    flag = " ★" if n >= 10 and acc >= 0.70 else ""
    print(f"{name:<30} {n:>6} {100*acc:>7.1f}% {sum(h['correct'] for h in hits):>9}{flag}")

# ── Deep-dive on high-value patterns ──────────────────────────────────────────
print("\n\n" + "=" * 65)
print("DEEP DIVE: high-value patterns (≥10 hits, ≥70% acc)")
print("=" * 65)
for name, hits in results.items():
    n = len(hits)
    if n < 10:
        continue
    acc = sum(h["correct"] for h in hits) / n
    if acc < 0.70:
        continue
    print(f"\n{'='*40}")
    print(f"[{name}] n={n} acc={100*acc:.1f}%")
    print(f"{'='*40}")
    print("CORRECT samples:")
    for h in [x for x in hits if x["correct"]][:4]:
        tail = h["raw"][-300:] if len(h["raw"]) > 300 else h["raw"]
        print(f"  GT={h['gt']} PRED={h['pred']} match={repr(h['match'][:80])}")
        print(f"  ...{tail[-150:]!r}\n")
    wrongs = [x for x in hits if not x["correct"]]
    if wrongs:
        print("WRONG samples:")
        for h in wrongs[:3]:
            tail = h["raw"][-300:] if len(h["raw"]) > 300 else h["raw"]
            print(f"  GT={h['gt']} PRED={h['pred']} match={repr(h['match'][:80])}")
            print(f"  ...{tail[-150:]!r}\n")
