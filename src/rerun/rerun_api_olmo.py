import os
import json
import pandas as pd
import requests
import time

# Default model and keys
MODEL_ID = "openai/gpt-oss-20b"

# Load candidate API keys from config if available (try multiple if one is rate/limit blocked)
API_KEYS = []
try:
    with open(os.path.join('config','model_keys.json'), encoding='utf-8') as kf:
        keys = json.load(kf)
        for e in keys:
            mid = e.get('model_id','')
            alias = e.get('api_key_alias')
            if alias and ('gpt-oss-20b' in mid):
                API_KEYS.append(alias)
except Exception:
    API_KEYS = []

# Fallback single key if none found in config
if not API_KEYS:
    API_KEYS = ["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"]
# Increase max tokens for reruns as requested
MAX_TOKENS = 4096
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"  # change if different

def call_model_api(messages, temperature=0.0):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature
    }
    # Try keys in order until one succeeds or all fail
    last_err = None
    for key in API_KEYS:
        # Try a few common header names in case the provider expects a different header
        header_options = [
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"x-api-key": key, "Content-Type": "application/json"},
            {"OpenAI-Api-Key": key, "Content-Type": "application/json"}
        ]
        response = None
        for headers in header_options:
            try:
                print(f"Trying key (masked): {key[:8]}... with header {list(headers.keys())[0]}")
                response = requests.post(ENDPOINT, headers=headers, json=payload)
            except Exception as e:
                print("Request exception:", e)
                response = None
            if response is not None and response.status_code == 200:
                break
        if response is not None and response.status_code == 200:
            return response.json()
        elif response is not None:
            last_err = (response.status_code, response.text)
            print("API error:", response.status_code, response.text)
            # If 403 due to key limit, try next key; 401 may mean wrong header/key combination
            if response.status_code in (403, 429):
                continue
            else:
                # For 401/other errors try next key too
                continue
    # all keys failed
    if last_err:
        return None
    return None

def main():
    df = pd.read_csv("rerun_plan.csv")
    for idx, row in df.iterrows():
        json_path = row["json_path"]
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # Build messages: prefer stored request.messages, else reconstruct from input
            messages = None
            req = data.get("request", {})
            if req and req.get("messages"):
                messages = req.get("messages")
            else:
                inp = data.get("input", {})
                persona_prompt = inp.get("persona", {}).get("explicit_prompt", "")
                question = inp.get("question", "")
                user_content = (persona_prompt + " " + question + "\n\nAnswer:").strip()
                messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": user_content}]

            # Determine temperature if available
            temperature = req.get("parameters", {}).get("temperature", 0.0) if req else 0.0

            print(f"Rerun {json_path} ...")
            result = call_model_api(messages, temperature=temperature)
            if result:
                # Extract content robustly
                content = None
                try:
                    # OpenRouter-style: choices[0]['message']['content']
                    content = result.get("choices", [])[0].get("message", {}).get("content")
                except Exception:
                    pass
                if not content:
                    try:
                        # Some providers return choices[0]['text']
                        content = result.get("choices", [])[0].get("text")
                    except Exception:
                        content = None

                # Save full raw response and extracted content
                data.setdefault("response", {})
                data["response"]["raw_api_response"] = result
                data["response"]["content"] = content if content is not None else ""
                usage = result.get("usage") if isinstance(result, dict) else None
                if usage:
                    data["response"]["usage"] = usage

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                print(f"No response for {json_path}")
            time.sleep(1.2)  # avoid rate limit
        else:
            print(f"File tidak ditemukan: {json_path}")

if __name__ == "__main__":
    main()
