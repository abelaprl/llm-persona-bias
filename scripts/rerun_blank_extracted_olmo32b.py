import pandas as pd
import os
import time
import requests
import json

# === CONFIG ===
CSV_BLANK_PATH = r"E:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think\csv_blank_extracted.csv"
INPUT_ROOT = r"E:\Tugas Akhir\TugasAkhir\result\allenai_olmo-3-32b-think"
API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Ganti jika endpoint berbeda
API_KEY = "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"  # from model_keys.json
MODEL = "allenai/olmo-3-32b-think"
MAX_TOKENS = 4096
RETRY_DELAY = 3
REQUEST_DELAY = 2

# === LOAD BLANK ROWS ===
df = pd.read_csv(CSV_BLANK_PATH, dtype=str)

for idx, row in df.iterrows():
    persona = row["persona_folder"]
    relfile = row["file"]
    json_path = os.path.join(INPUT_ROOT, relfile)
    if not os.path.exists(json_path):
        print(f"[SKIP] File not found: {json_path}")
        continue
    # Load original JSON
    with open(json_path, encoding="utf-8") as jf:
        data = json.load(jf)
    # Prepare API request
    prompt = data["input"]["question"] if "input" in data and "question" in data["input"] else None
    if not prompt:
        print(f"[SKIP] No prompt in: {json_path}")
        continue
    messages = [{"role": "user", "content": prompt}]
    req = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": data.get("request", {}).get("parameters", {}).get("temperature", 0.0),
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # Retry logic
    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=req, timeout=120)
            if resp.status_code == 200:
                api_resp = resp.json()
                # Update response in JSON
                data["response"] = {
                    "raw_api_response": api_resp,
                    "content": None  # Let pipeline extract as usual
                }
                # Optionally update request max_tokens
                if "request" in data:
                    data["request"]["parameters"]["max_tokens"] = MAX_TOKENS
                # Save back to file
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(data, jf, ensure_ascii=False, indent=2)
                print(f"[OK] {relfile}")
                break
            else:
                print(f"[ERR] {relfile} HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[ERR] {relfile} Exception: {e}")
        time.sleep(RETRY_DELAY)
    time.sleep(REQUEST_DELAY)
print("Done rerun blank extracted_answer.")
