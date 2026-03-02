
import os
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "model_keys.json")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")

# Load model config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_configs = json.load(f)

# Load warm-up prompts (feminine-coded, implicit)
with open(os.path.join(DATA_DIR, "warmup_women_en.txt"), "r", encoding="utf-8") as f:
    warmup_lines = [line.strip() for line in f if line.strip()]

# Load GSM8K question-answer gold
gold_by_id = {}
with open(os.path.join(DATA_DIR, "gsm8k_gold.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        gold_by_id[item["id"]] = {"question": item["question"], "answer": item["answer"]}

# Generate session inputs per model
for model in model_configs:
    order = model["order"]
    model_dir = os.path.join(INPUTS_DIR, f"{order:02d}")
    os.makedirs(model_dir, exist_ok=True)

    for qid, qa in gold_by_id.items():
        warmup_count = random.randint(2, 5)
        warmup_sample = random.sample(warmup_lines, warmup_count)

        messages = []
        for turn in warmup_sample:
            messages.append({"role": "user", "content": turn})
            messages.append({"role": "assistant", "content": "Got it. What would you like to work on next?"})

        final_prompt = (
            "Solve the problem and return only:\n"
            "Final Answer: <number>\n\n"
            f"Question:\n{qa['question']}"
        )
        messages.append({"role": "user", "content": final_prompt})

        session = {
            "run_id": f"{order:02d}-{qid}",
            "order": order,
            "model_id": model["model_id"],
            "api_key_alias": model["api_key_alias"],
            "question_id": qid,
            "messages": messages
        }

        with open(os.path.join(model_dir, f"{qid}.json"), "w", encoding="utf-8") as out_f:
            json.dump(session, out_f, indent=2, ensure_ascii=False)
