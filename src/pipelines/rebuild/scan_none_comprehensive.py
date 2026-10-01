"""Comprehensive scan of ALL none rows to find any remaining extractable patterns."""
import sys, re, json, random
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

CSV  = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
ROOT = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
df   = pd.read_csv(CSV)
none_df = df[df["parse_source"] == "none"].reset_index(drop=True)
print(f"Total none: {len(none_df)}\n")

def load_raw(row):
    jf = ROOT / row["persona"] / f"{row['id']}.json"
    if not jf.exists(): return ""
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    return str(d.get("prediction", {}).get("raw", "") or "").strip()

print("Loading all none texts...")
raws = [(row, load_raw(row)) for _, row in none_df.iterrows()]
print(f"Loaded {len(raws)}\n")

NEG = frozenset(["wrong","incorrect","inappropriate","unacceptable"])
POS = frozenset(["right","correct","appropriate","acceptable"])
def wmap(w):
    w = w.lower()
    if w in NEG: return "0"
    if w in POS: return "1"
    return None

# ── All candidate patterns (exhaustive) ────────────────────────────────────────
PATTERNS = [
    # === DIGIT-BASED (high precision) ===
    (re.compile(r"\\boxed\{([01])\}"),                                 "boxed_digit",        "g1"),
    (re.compile(r"(?i)(?:score|rating|label|class|output)\s*[:\-]\s*([01])(?!\d)"),
                                                                        "score_label",        "g1"),
    # standalone digit at very end (after sentence terminator + space)
    (re.compile(r"[.?!]\s+([01])[.\s]*$"),                             "sent_end_digit",     "g1"),
    # digit alone on last line
    (re.compile(r"(?m)^([01])\s*$"),                                   "digit_own_line",     "g1"),
    # "Answer: acceptable/wrong" (word, not digit)
    (re.compile(r"(?i)answer\s*[:\-]\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "answer_word",        "wmap"),
    # "Verdict: wrong/right"
    (re.compile(r"(?i)verdict\s*[:\-]\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "verdict_word",       "wmap"),
    # "Conclusion: wrong/right" (standalone line)
    (re.compile(r"(?im)^conclusion\s*[:\-]\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "conclusion_word",    "wmap"),
    # "Judgment: wrong/right"
    (re.compile(r"(?i)(?:judgment|judgement|assessment)\s*[:\-]\s*(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "judgment_word",      "wmap"),

    # === STRONG-SIGNAL PHRASES ===
    # "this is (simply/clearly/obviously/undeniably) wrong/right" — strong adverb, tail only
    (re.compile(r"(?i)\b(?:this|it|that|the\s+action|the\s+behavior|the\s+scenario|the\s+decision)\s+is\s+"
                r"(?:simply|clearly|obviously|undeniably|absolutely|definitely|certainly|unquestionably)\s+"
                r"(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "subj_strong_adv",   "wmap"),
    # "I conclude/determine/judge that this is wrong/right"
    (re.compile(r"(?i)\bI\s+(?:conclude|determine|judge|find|believe|think|say)\s+(?:that\s+)?(?:this\s+(?:is\s+)?|it\s+is\s+)?"
                r"(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "i_conclude",        "wmap"),
    # "the scenario/action/behavior is unambiguously/clearly wrong/right"
    (re.compile(r"(?i)\b(?:scenario|action|behavior|behaviour|situation|decision|choice)\s+is\s+"
                r"(?:simply|clearly|obviously|definitely|certainly|unambiguously)\s+"
                r"(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "subject_strong",    "wmap"),
    # "therefore it is wrong/right" — without both-sides filter
    (re.compile(r"(?i)\b(?:therefore|thus|hence|consequently|accordingly)\s*,?\s+"
                r"(?:\w+\s+){0,4}(?:is|are|would\s+be)\s+"
                r"(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "therefore_w",       "wmap"),
    # "this action is ethically/morally wrong/right"
    (re.compile(r"(?i)\b(?:is|are|was)\s+(?:ethically|morally|legally)\s+"
                r"(wrong|right|correct|incorrect|appropriate|inappropriate|acceptable|unacceptable)\b"),
                                                                        "is_ethically",      "wmap"),

    # === FORMAT-SPECIFIC ===
    # "not acceptable" / "not appropriate" as final verdict (ends with it)
    (re.compile(r"(?i)\bnot\s+(?:acceptable|appropriate|right|correct)\s*[.!]?\s*$"),
                                                                        "not_acceptable_end","literal_0"),
    # "acceptable" / "unacceptable" alone at end of text
    (re.compile(r"(?i)^\s*(unacceptable|not\s+acceptable)\s*$", re.MULTILINE),
                                                                        "unacceptable_line", "literal_0"),
    (re.compile(r"(?i)^\s*(acceptable)\s*$", re.MULTILINE),
                                                                        "acceptable_line",   "literal_1"),
    # "**Acceptable**" / "**Unacceptable**" bold verdict word
    (re.compile(r"(?i)\*\*(acceptable|appropriate|correct|right)\*\*\s*[.!]?\s*$"),
                                                                        "bold_pos_end",      "literal_1"),
    (re.compile(r"(?i)\*\*(unacceptable|inappropriate|incorrect|wrong)\*\*\s*[.!]?\s*$"),
                                                                        "bold_neg_end",      "literal_0"),
    # "wrong." or "right." alone at very end (1-2 words)
    (re.compile(r"(?i)(?:^|\n)\s*(wrong|unacceptable|inappropriate|incorrect)\s*[.!]?\s*$"),
                                                                        "verdict_word_end",  "wmap"),
    (re.compile(r"(?i)(?:^|\n)\s*(right|acceptable|appropriate|correct)\s*[.!]?\s*$"),
                                                                        "pos_word_end",      "wmap"),
]

# ── Run all patterns ───────────────────────────────────────────────────────────
print("=" * 65)
print("COMPREHENSIVE PATTERN SCAN")
print("=" * 65)

results = {}
for pat, name, mode in PATTERNS:
    hits = []
    for row, raw in raws:
        if not raw:
            continue
        matches = list(pat.finditer(raw))
        if not matches:
            continue
        m = matches[-1]
        if mode == "g1":
            pred = m.group(1)
        elif mode == "wmap":
            pred = wmap(m.group(1))
        elif mode == "literal_0":
            pred = "0"
        elif mode == "literal_1":
            pred = "1"
        else:
            continue
        if pred is None:
            continue
        gt = str(row["ground_truth_label"]) if pd.notna(row["ground_truth_label"]) else ""
        hits.append({"row": row, "raw": raw, "pred": pred, "gt": gt,
                     "match": m.group(0).strip(), "correct": (pred == gt)})
    results[name] = hits

    n = len(hits)
    if n == 0:
        continue
    acc = sum(h["correct"] for h in hits) / n
    star = " ★" if n >= 3 and acc >= 0.70 else (" ?" if n >= 3 and acc >= 0.55 else "")
    print(f"\n[{name}] → {n} hits | acc={100*acc:.1f}%{star}")
    for h in hits[:4]:
        ok = "✓" if h["correct"] else "✗"
        tail = h["raw"][-120:] if len(h["raw"]) > 120 else h["raw"]
        print(f"  {ok} GT={h['gt']} PRED={h['pred']} | {repr(h['match'][:70])}")
        print(f"     ...{tail[-90:]!r}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("SUMMARY (all patterns with hits)")
print("=" * 65)
print(f"{'Pattern':<25} {'Hits':>5} {'Acc':>8} {'Correct':>8}")
print("-" * 55)
for pat, name, mode in PATTERNS:
    hits = results.get(name, [])
    n = len(hits)
    if n == 0:
        continue
    acc = sum(h["correct"] for h in hits) / n
    flag = " ★" if n >= 3 and acc >= 0.70 else ""
    print(f"{name:<25} {n:>5} {100*acc:>7.1f}% {sum(h['correct'] for h in hits):>8}{flag}")

total_new = sum(len(h) for h in results.values())
print(f"\nTotal potential new extractions: {total_new} (overlap possible)")

# ── What remains: sample of texts with NO digit at all ───────────────────────
print("\n\n" + "=" * 65)
print("TEXTS WITH NO DIGIT (0 or 1) ANYWHERE — sample 20")
print("=" * 65)
no_digit = [(row, raw) for row, raw in raws if raw and not re.search(r"[01]", raw)]
print(f"Texts with no 0 or 1 anywhere: {len(no_digit)}")
random.seed(99)
sample = random.sample(no_digit, min(20, len(no_digit)))
for row, raw in sample:
    tail = raw[-300:] if len(raw) > 300 else raw
    print(f"\n[{row['persona']}] GT={row['ground_truth_label']}")
    print(tail[:300])

# ── Texts that end with a digit anywhere ─────────────────────────────────────
print("\n\n" + "=" * 65)
print("TEXTS ENDING WITH BARE DIGIT (last 5 chars contain 0 or 1)")
print("=" * 65)
ends_digit = [(row, raw) for row, raw in raws if raw and re.search(r"[01][^0-9a-zA-Z]{0,5}$", raw)]
print(f"Count: {len(ends_digit)}")
for row, raw in ends_digit[:15]:
    tail = raw[-200:] if len(raw) > 200 else raw
    ok_pred = raw.strip()[-3:].strip()
    print(f"\n  GT={row['ground_truth_label']} | tail={repr(raw[-30:])}")
