from datasets import load_dataset
import os, json

# 1. Load full MMLU (all subjects, test split)
print("⬇️ Loading cais/mmlu (config='all', split='test') from HuggingFace...")
dataset = load_dataset("cais/mmlu", "all", split="test")

print("✅ Loaded", len(dataset), "rows")

# 2. Simpan ke format mirip mmlu_raw.json yang tadi kita pakai
# Simpan ke folder `data/` di root repository (bukan `scripts/data`)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

out_path = os.path.join(DATA_DIR, "mmlu_raw.json")

rows = []
for i, ex in enumerate(dataset):
    # ex berisi: question, subject, choices, answer, etc.
    rows.append({"row": ex, "row_idx": i})

with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"rows": rows}, f, ensure_ascii=False)

print(f"💾 Saved full MMLU raw to: {out_path}")
print("   Contoh keys dalam satu row:", list(dataset[0].keys()))
