import json
import os
import requests


def load_model_config(config_path: str, name: str):
    with open(config_path, "r", encoding="utf-8") as f:
        models = json.load(f)
    for m in models:
        if m.get("name") == name:
            return m
    raise ValueError(f"Model '{name}' not found in {config_path}")


def chat(messages, api_key: str, endpoint_url: str, model_id: str, temperature: float = 0.2):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_text(resp_json):
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json)
