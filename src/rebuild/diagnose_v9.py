import pandas as pd, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

csv_path = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
df = pd.read_csv(csv_path)

# === 1. Accuracy per parse_source ===
print("=== ACCURACY PER parse_source ===")
scored = df[df["parsed_answer"].isin(["0", "1"]) | df["parsed_answer"].isin([0, 1])]
# is_correct is True/False/""
scored = df[df["is_correct"].isin([True, False])]
for src, grp in scored.groupby("parse_source"):
    n = len(grp)
    correct = grp["is_correct"].sum()
    print(f"  {src:25s}: n={n:5d}  acc={100*correct/n:.1f}%")

print(f"\n  {'TOTAL':25s}: n={len(scored):5d}  acc={100*scored['is_correct'].sum()/len(scored):.2f}%")

# === 2. Sample 20 semantic_is cases and show raw ===
print("\n=== SAMPLE semantic_is CASES (10 correct, 10 wrong) ===")
sem = scored[scored["parse_source"] == "semantic_is"].copy()
correct_sem = sem[sem["is_correct"] == True].sample(min(10, len(sem[sem["is_correct"]==True])), random_state=1)
wrong_sem   = sem[sem["is_correct"] == False].sample(min(10, len(sem[sem["is_correct"]==False])), random_state=1)

root = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")

def show_sample(row, label):
    jf = root / row["persona"] / f"{row['id']}.json"
    if not jf.exists():
        return
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    raw = str(d.get("prediction", {}).get("raw", "") or "").strip()
    tail = raw[-300:] if len(raw) > 300 else raw
    print(f"\n[{label}] GT={row['ground_truth_label']} PRED={row['prediction_label']} | {row['persona']}")
    print(tail)

print("\n--- CORRECT ---")
for _, r in correct_sem.iterrows():
    show_sample(r, "OK")

print("\n--- WRONG (false positive) ---")
for _, r in wrong_sem.iterrows():
    show_sample(r, "FP")

# === 3. Sample 20 remaining none ===
print("\n\n=== SAMPLE remaining none (20 cases) ===")
none_df = df[df["parse_source"] == "none"].sample(20, random_state=77)
for _, row in none_df.iterrows():
    jf = root / row["persona"] / f"{row['id']}.json"
    if not jf.exists():
        print(f"\n[MISSING] {row['id']} {row['persona']}")
        continue
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    raw = str(d.get("prediction", {}).get("raw", "") or "").strip()
    tail = raw[-300:] if len(raw) > 300 else raw
    print(f"\n--- [{row['persona']}] GT={row['ground_truth_label']} ---")
    print(tail)
