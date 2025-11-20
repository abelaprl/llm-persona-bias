import os
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
MMLU_SUBFOLDER = "mmlu"
os.makedirs(os.path.join(INPUTS_DIR, MMLU_SUBFOLDER), exist_ok=True)

# Load model config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_configs = json.load(f)

# Load warm-up lines
with open(os.path.join(DATA_DIR, "warmup_women_en.txt"), "r", encoding="utf-8") as f:
    warmup_lines = [line.strip() for line in f if line.strip()]

# Load MMLU-style dataset (you must fill this file with real MMLU samples)
mmlu_path = os.path.join(DATA_DIR, "mmlu_gold.jsonl")
mmlu_questions = []
with open(mmlu_path, "r", encoding="utf-8") as f:
    for line in f:
        mmlu_questions.append(json.loads(line))

# Generate session inputs per model for MMLU
for model in model_configs:
    order = model["order"]
    model_dir = os.path.join(INPUTS_DIR, MMLU_SUBFOLDER, f"{order:02d}")
    os.makedirs(model_dir, exist_ok=True)

    for idx, q in enumerate(mmlu_questions):
        qid = q["id"]
        prompt = (
            f"Choose the correct answer and respond with only:\nFinal Answer: <A/B/C/D>\n\n"
            f"Question:\n{q['question']}\n"
            f"A. {q['A']}\nB. {q['B']}\nC. {q['C']}\nD. {q['D']}"
        )

        warmup_count = random.randint(2, 5)
        warmup_sample = random.sample(warmup_lines, warmup_count)

        messages = []
        for turn in warmup_sample:
            messages.append({"role": "user", "content": turn})
            messages.append({"role": "assistant", "content": "Noted. What's your next task?"})

        messages.append({"role": "user", "content": prompt})

        session = {
            "run_id": f"mmlu-{order:02d}-{qid}",
            "order": order,
            "task": "mmlu",
            "model_id": model["model_id"],
            "api_key_alias": model["api_key_alias"],
            "question_id": qid,
            "messages": messages
        }

        with open(os.path.join(model_dir, f"{qid}.json"), "w", encoding="utf-8") as out_f:
            json.dump(session, out_f, indent=2, ensure_ascii=False)