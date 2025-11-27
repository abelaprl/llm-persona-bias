import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# === Parse GSM8K ===
gsm8k_raw = os.path.join(DATA_DIR, "gsm8k_raw.json")
gsm8k_out = os.path.join(DATA_DIR, "gsm8k_gold.jsonl")

if os.path.exists(gsm8k_raw):
    with open(gsm8k_raw, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]

    with open(gsm8k_out, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            d = row["row"]
            qid = d.get("id", f"gsm8k_{i}")
            question = d.get("question", "")
            answer = d.get("answer", "")
            out.write(json.dumps({"id": qid, "question": question, "answer": answer}) + "\n")

    print(f"✅ GSM8K parsed → {gsm8k_out}")
else:
    print("⚠️ GSM8K raw file not found!")


# === Parse MMLU ===
mmlu_raw = os.path.join(DATA_DIR, "mmlu_raw.json")
mmlu_out = os.path.join(DATA_DIR, "mmlu_gold.jsonl")

if os.path.exists(mmlu_raw):
    with open(mmlu_raw, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]

    with open(mmlu_out, "w", encoding="utf-8") as out:
        for row in rows:
            d = row["row"]
            qid = d.get("id", f"mmlu_{row['row_idx']}")
            question = d.get("question", "")
            choices = d.get("choices", [])
            answer = d.get("answer", None)

            record = {
                "id": qid,
                "question": question,
                "A": choices[0] if len(choices) > 0 else "",
                "B": choices[1] if len(choices) > 1 else "",
                "C": choices[2] if len(choices) > 2 else "",
                "D": choices[3] if len(choices) > 3 else "",
                "answer": answer,
            }
            out.write(json.dumps(record) + "\n")

    print(f"✅ MMLU parsed → {mmlu_out}")
else:
    print("⚠️ MMLU raw file not found!")
