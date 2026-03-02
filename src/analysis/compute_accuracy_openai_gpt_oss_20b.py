import csv
import json

# Paths
CSV_PATH = 'apply_extractor_openai_gpt_oss_20b.csv'
GOLD_PATH = 'data/gsm8k_test.json'

# Load gold answers
def load_gold_answers():
    with open(GOLD_PATH, encoding='utf-8') as f:
        gold = json.load(f)
    # Map: filename (gsm8k_XXXXX.json) -> gold answer (number as string)
    gold_map = {}
    for i, item in enumerate(gold):
        # Extract gold answer from 'answer' field (#### number)
        import re
        m = re.search(r'####\s*([-+]?[$€£]?\s*[\d,]+(?:\.\d+)?)', item['answer'])
        if m:
            gold_map[f"gsm8k_{i:05d}.json"] = m.group(1).replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    return gold_map

def main():
    gold = load_gold_answers()
    total = 0
    correct = 0
    missing = 0
    with open(CSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['file']
            pred = row['extracted_answer']
            gold_ans = gold.get(fname)
            if gold_ans is None:
                missing += 1
                continue
            total += 1
            # Compare as float for robustness
            try:
                if pred is not None and pred != '' and float(pred) == float(gold_ans):
                    correct += 1
            except Exception:
                pass
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.4f}")
    if missing:
        print(f"Missing gold answers for {missing} files")

if __name__ == '__main__':
    main()
