import pandas as pd, json
from pathlib import Path

csv_path = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
df = pd.read_csv(csv_path)
none_df = df[df["parse_source"] == "none"].sample(80, random_state=99)
row = none_df.iloc[29]  # sample #30

print(f"Persona: {row['persona']}, ID: {row['id']}, GT: {row['ground_truth_label']}")

root = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
jf = root / row["persona"] / f"{row['id']}.json"
with open(jf, encoding="utf-8") as f:
    d = json.load(f)

raw = d.get("prediction", {}).get("raw", "") or ""
print("\n=== FULL RAW ===")
print(raw)
print(f"\n=== SCENARIO ===")
print(d.get("scenario", ""))
