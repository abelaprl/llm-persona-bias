"""Inspect all none rows that end with a digit — find new precise patterns."""
import sys, re, json
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CSV  = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
ROOT = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
df   = pd.read_csv(CSV)
none_df = df[df["parse_source"] == "none"].reset_index(drop=True)

def load_raw(row):
    jf = ROOT / row["persona"] / f"{row['id']}.json"
    if not jf.exists(): return ""
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    return str(d.get("prediction", {}).get("raw", "") or "").strip()

# All none rows ending with a digit (last 10 chars)
print("=== ALL NONE ROWS ENDING WITH 0 or 1 (last 10 chars) ===\n")
count = 0
for _, row in none_df.iterrows():
    raw = load_raw(row)
    if not raw:
        continue
    if re.search(r"[01][^0-9a-zA-Z]{0,5}$", raw):
        count += 1
        tail = raw[-500:] if len(raw) > 500 else raw
        print(f"--- #{count} GT={row['ground_truth_label']} | {row['persona']} | {row['id']} ---")
        print(tail)
        print()

print(f"\nTotal ending with digit: {count}")

# Also look for texts with "**Response" or "**Binary" patterns
print("\n\n=== TEXTS WITH **Response** or **Binary Response** PATTERN ===\n")
pat_resp = re.compile(r"(?i)\*\*\s*(?:response|binary\s+response)[^*]{0,80}\*\*")
count2 = 0
for _, row in none_df.iterrows():
    raw = load_raw(row)
    if not raw:
        continue
    if pat_resp.search(raw):
        count2 += 1
        tail = raw[-400:] if len(raw) > 400 else raw
        print(f"--- #{count2} GT={row['ground_truth_label']} | {row['persona']} ---")
        print(tail)
        print()
print(f"Total: {count2}")

# Texts with "score this scenario" or "I will provide"
print("\n\n=== TEXTS WITH 'score this scenario' or 'I will provide' ===\n")
pat_score = re.compile(r"(?i)(?:score\s+this\s+scenario|i\s+will\s+provide|providing\s+a\s+(?:binary|numeric)|response\s+of\s+[01]|regarding\s+the\s+scenario\s*:)")
count3 = 0
for _, row in none_df.iterrows():
    raw = load_raw(row)
    if not raw:
        continue
    m = pat_score.search(raw)
    if m:
        count3 += 1
        tail = raw[-400:] if len(raw) > 400 else raw
        print(f"--- #{count3} GT={row['ground_truth_label']} | {row['persona']} | match={repr(m.group(0))} ---")
        print(tail)
        print()
print(f"Total: {count3}")
