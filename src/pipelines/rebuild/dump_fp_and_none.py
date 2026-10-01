"""Dump semantic_is FP cases and remaining none cases to txt files."""
import sys, re, json
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CSV  = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
ROOT = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
df   = pd.read_csv(CSV)

def load_raw(row):
    jf = ROOT / row["persona"] / f"{row['id']}.json"
    if not jf.exists():
        return ""
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    return str(d.get("prediction", {}).get("raw", "") or "").strip()

# ── 1. semantic_is FALSE POSITIVES (all of them) ──────────────────────────────
sem_df = df[df["parse_source"] == "semantic_is"].copy()
fp_df  = sem_df[sem_df["is_correct"] == False].reset_index(drop=True)

lines_fp = [
    f"semantic_is FALSE POSITIVES — {len(fp_df)} cases",
    f"(GT≠PRED, parse_source=semantic_is)",
    "=" * 70,
    "",
]
for i, (_, row) in enumerate(fp_df.iterrows(), 1):
    raw  = load_raw(row)
    tail = raw[-500:] if len(raw) > 500 else raw
    lines_fp.append(f"--- {i} | {row['persona']} | GT={row['ground_truth_label']} PRED={row['prediction_label']} | {row['id']} ---")
    lines_fp.append(tail)
    lines_fp.append("")

Path("semantic_is_fp.txt").write_text("\n".join(lines_fp), encoding="utf-8")
print(f"Written {len(fp_df)} FP cases to semantic_is_fp.txt")

# ── 2. Remaining NONE (all 2605) ───────────────────────────────────────────────
none_df = df[df["parse_source"] == "none"].reset_index(drop=True)

lines_none = [
    f"NONE (unparsed) — {len(none_df)} cases",
    "=" * 70,
    "",
]
for i, (_, row) in enumerate(none_df.iterrows(), 1):
    raw  = load_raw(row)
    tail = raw[-500:] if len(raw) > 500 else raw
    lines_none.append(f"--- {i} | {row['persona']} | GT={row['ground_truth_label']} | {row['id']} ---")
    lines_none.append(tail)
    lines_none.append("")

Path("none_all.txt").write_text("\n".join(lines_none), encoding="utf-8")
print(f"Written {len(none_df)} none cases to none_all.txt")
