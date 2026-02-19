import os
import json
import pandas as pd
import requests
import time

API_KEY = "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"
MODEL_ID = "allenai/olmo-3-32b-think"
MAX_TOKENS = 4096
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"  # Ganti jika endpoint berbeda

def call_model_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print("API error:", response.status_code, response.text)
        return None

def main():
    df = pd.read_csv("rerun_plan.csv")
    for idx, row in df.iterrows():
        json_path = row["json_path"]
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("question", "")
            if not prompt:
                print(f"Prompt kosong untuk {json_path}")
                continue
            print(f"Rerun {json_path} ...")
            result = call_model_api(prompt)
            if result and "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                data["response_content"] = content
                data["completion_tokens"] = result.get("usage", {}).get("completion_tokens", None)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            time.sleep(1.2)  # Hindari rate limit
        else:
            print(f"File tidak ditemukan: {json_path}")

if __name__ == "__main__":
    main()
