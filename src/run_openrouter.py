from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import requests

from src.load_keys import load_model_keys, get_models, find_model_by_name


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_prompt_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def call_openrouter(model_id: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_text(response_json: Dict[str, Any]) -> str:
    try:
        choices = response_json.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
    except Exception:
        pass
    return ""


def main():
    # 1) Load models
    keys = load_model_keys()
    models = get_models(keys)

    # 2) Pick model (adjust name as needed)
    model = find_model_by_name(models, "polaris-alpha") or models[0]
    model_id = model.get("model_id")
    api_key = model.get("api_key_alias")

    if not api_key:
        raise RuntimeError("api_key_alias missing for selected model. Please set it in config/model_keys.json")

    # 3) Load prompt
    prompt_path = "inputs/implicits_woman_promt.json"  # adjust as needed
    prompt_json = load_prompt_file(prompt_path)
    system_prompt = prompt_json.get("system", "You are a helpful assistant.")
    user_prompt = prompt_json.get("user", "Tolong jawab dengan penalaran langkah demi langkah.")

    # 4) Call API
    start = time.time()
    try:
        response_json = call_openrouter(model_id, api_key, system_prompt, user_prompt)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Response text: {getattr(e.response, 'text', '')}")
        return
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
        return
    latency = time.time() - start

    # 5) Extract and print result safely
    text = extract_text(response_json)
    print("Model:", model_id)
    print("Latency (sec):", round(latency, 2))
    print("Output (first 500 chars):")
    print(text[:500])


if __name__ == "__main__":
    main()
