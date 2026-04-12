#!/usr/bin/env python3
import requests, json, os
from datetime import datetime

MODEL_ID, MODEL_NAME = "qwen/qwen3-next-80b-a3b-thinking", "qwen_qwen3-next-80b-a3b-thinking"
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE, MAX_TOKENS, MAX_RETRIES = 0.2, 512, 5

with open("inputs/explicits_man_prompts.json", "r", encoding="utf-8") as f:
    persona = next(p for p in json.load(f) if p['id'] == 1)
with open("data/gsm8k_test.json", "r", encoding="utf-8") as f:
    questions = json.load(f)[:5]

results_dir = f"results/{MODEL_NAME}/man_explicit"
os.makedirs(results_dir, exist_ok=True)

def call_api(messages, attempt=1):
    try:
        r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {API_KEY}"}, 
                         json={"model": MODEL_ID, "messages": messages, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}, timeout=30)
        if r.status_code == 200:
            res = r.json()
            return {"success": True, "content": res['choices'][0]['message']['content'], "raw_response": res, "latency_sec": r.elapsed.total_seconds(), "attempts": attempt}
        elif attempt < MAX_RETRIES:
            return call_api(messages, attempt + 1)
        else:
            return {"success": False, "error": r.json().get('error', {}).get('message', f'HTTP {r.status_code}'), "attempts": attempt}
    except Exception as e:
        return call_api(messages, attempt + 1) if attempt < MAX_RETRIES else {"success": False, "error": str(e), "attempts": attempt}

def create_output(q_data, q_id, api_resp):
    return {
        "metadata": {"run_id": f"{MODEL_NAME}_man_explicit_{persona['id']:03d}_{q_id:05d}", "experiment_type": "gsm8k_bias_study", 
                    "timestamp": datetime.now().isoformat(), "model_info": {"provider": "openrouter", "model_id": MODEL_ID, "api_endpoint": OPENROUTER_URL}},
        "input": {"question_id": q_id, "dataset": "gsm8k_test", "question": q_data["question"], "ground_truth": q_data["answer"],
                 "persona": {"id": persona["id"], "type": "man_explicit", "prompt": persona["prompt"]}},
        "request": {"messages": [{"role": "system", "content": persona["prompt"]}, {"role": "user", "content": q_data["question"]}],
                   "parameters": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "model": MODEL_ID}},
        "response": {"success": api_resp.get("success"), "content": api_resp.get("content"), "raw_api_response": api_resp.get("raw_response"),
                    "error": api_resp.get("error"), "performance": {"latency_sec": api_resp.get("latency_sec"), 
                    "tokens": api_resp.get("raw_response", {}).get("usage") if api_resp.get("raw_response") else None, "attempts": api_resp.get("attempts")}},
        "analysis": {"extracted_answer": None, "is_correct": None, "reasoning_quality": None, "bias_indicators": None}
    }

print(f"\n{'='*80}\nTesting: {MODEL_ID}\n{'='*80}\n")
success_count = 0
for idx, q_data in enumerate(questions, 1):
    q_id = q_data.get("id", idx)
    print(f"[{idx}] Q{q_id}...", end=" ")
    api_resp = call_api([{"role": "system", "content": persona["prompt"]}, {"role": "user", "content": q_data["question"]}])
    if api_resp.get("success"):
        print(f"✅ ({api_resp.get('latency_sec', 0):.2f}s)")
        success_count += 1
    else:
        print(f"❌ {api_resp.get('error')}")
    with open(f"{results_dir}/gsm8k_{idx:05d}.json", 'w', encoding='utf-8') as f:
        json.dump(create_output(q_data, q_id, api_resp), f, indent=2, ensure_ascii=False)
print(f"{'='*80}\n✅ {success_count}/5 successful\n{'='*80}")
