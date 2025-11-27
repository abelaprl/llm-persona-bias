from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.load_keys import load_model_keys, get_models, find_model_by_name


def load_prompt_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_request_payload(model_id: str, api_key_alias: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Build a generic request payload example (does NOT send).
    Designed to be adapted for different providers (OpenRouter, Groq, etc.).
    """
    return {
        "model": model_id,
        "auth": {
            "api_key_alias": api_key_alias,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "text"},
        "metadata": {"dry_run": True},
    }


def main():
    # 1) Load models from config
    keys = load_model_keys()
    models = get_models(keys)

    # 2) Pick a model by name (change as needed)
    model = find_model_by_name(models, "polaris-alpha") or models[0]
    model_id = model.get("model_id")
    api_key_alias = model.get("api_key_alias")

    # 3) Load a prompt (implicit woman prompt as example)
    prompt_path = "inputs/implicits_woman_promt.json"  # adjust if needed
    prompt_json = load_prompt_file(prompt_path)

    # Expect JSON like: {"system": "...", "user": "..."}
    system_prompt = prompt_json.get("system", "You are a helpful assistant.")
    user_prompt = prompt_json.get("user", "Tolong jawab dengan penalaran langkah demi langkah.")

    # 4) Build a request payload (not sent)
    payload = build_request_payload(model_id, api_key_alias, system_prompt, user_prompt)

    # 5) Print summary (redact secret prefixes only)
    redacted_key = (str(api_key_alias)[:6] + "...") if api_key_alias else None
    summary = {
        "model_id": model_id,
        "api_key_alias": redacted_key,
        "prompt_file": prompt_path,
        "system_prompt_preview": system_prompt[:80],
        "user_prompt_preview": user_prompt[:80],
    }
    print("Runner summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDry-run payload (truncated):")
    # Truncate messages for display
    display_payload = {
        **payload,
        "messages": [
            {"role": "system", "content": system_prompt[:120]},
            {"role": "user", "content": user_prompt[:120]},
        ],
    }
    print(json.dumps(display_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
