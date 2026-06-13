import pandas as pd, json
from pathlib import Path

csv_path = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
df = pd.read_csv(csv_path)
none_df = df[df["parse_source"] == "none"].sample(120, random_state=55)
root = Path("result/ETHICS/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")

out = []
for i, (_, row) in enumerate(none_df.iterrows(), 1):
    jf = root / row["persona"] / f"{row['id']}.json"
    if not jf.exists():
        continue
    with open(jf, encoding="utf-8") as f:
        d = json.load(f)
    raw = str(d.get("prediction", {}).get("raw", "") or "").strip()
    tail = raw[-400:] if len(raw) > 400 else raw
    out.append(f"--- {i} [{row['persona']}] ---\n{tail}\n")

Path("none_samples_tmp.txt").write_text("\n".join(out), encoding="utf-8")
print(f"Written {len(out)} samples to none_samples_tmp.txt")
