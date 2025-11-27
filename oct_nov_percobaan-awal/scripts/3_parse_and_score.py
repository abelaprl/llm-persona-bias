import os
import json
import re
import csv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_json")
DATA_DIR = os.path.join(BASE_DIR, "data")
SUMMARY_PATH = os.path.join(BASE_DIR, "results", "summary")
os.makedirs(SUMMARY_PATH, exist_ok=True)

# Load gold answers
gold_answers = {}
with open(os.path.join(DATA_DIR, "gsm8k_gold.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qid = item["id"]
        match = re.search(r"####\\s*(\\-?\\d+(\\.\\d+)?)", item["answer"])
        if match:
            gold_answers[qid] = match.group(1).strip()

# Parse final answer from response text
def extract_final_answer(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"Final Answer:\\s*(\\-?\\d+(\\.\\d+)?)", text)
    if match:
        return match.group(1).strip()
    return None

# Collect scores per model
summary_rows = []
for order_folder in sorted(os.listdir(LOGS_DIR)):
    folder_path = os.path.join(LOGS_DIR, order_folder)
    if not os.path.isdir(folder_path):
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    n_total, n_correct, n_format = 0, 0, 0
    latencies = []

    for fname in files:
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            result = json.load(f)

        qid = result["question_id"]
        resp = result.get("response", {})
        latency = result.get("latency_sec", 0)
        latencies.append(latency or 0)

        # Try to parse model output
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        final = extract_final_answer(content)

        # Score
        n_total += 1
        if final is not None:
            n_format += 1
            if qid in gold_answers and final == gold_answers[qid]:
                n_correct += 1

    acc = n_correct / n_total if n_total else 0
    fmt = n_format / n_total if n_total else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    summary_rows.append({
        "model_order": order_folder,
        "total": n_total,
        "exact_match": n_correct,
        "format_ok": n_format,
        "acc": round(acc, 3),
        "format_rate": round(fmt, 3),
        "avg_latency_sec": round(avg_lat, 2)
    })

# Write CSV summary
csv_path = os.path.join(SUMMARY_PATH, "results_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"✅ Results summary written to: {csv_path}")