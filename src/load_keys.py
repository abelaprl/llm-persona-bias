from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_model_keys(path: str = "config/model_keys.json") -> Any:
    """
    Load model keys/config from `config/model_keys.json`.

    Supports two formats:
    1) Array of model entries (current repo format):
       [
         {"order": 0, "name": "...", "model_id": "...", "api_key_alias": "..."},
         ...
       ]

    2) Object with providers + models (future-friendly):
       {
         "providers": { "openrouter": {"api_key": "..."}, ... },
         "models": [ {"order": 0, "name": "...", "model_id": "...", "api_key_alias": "..."} ]
       }

    Returns the parsed JSON object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing credentials file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_models(keys: Any) -> List[Dict[str, Any]]:
    """
    Normalize to a list of model dicts.
    """
    if isinstance(keys, list):
        return keys
    if isinstance(keys, dict):
        models = keys.get("models")
        if isinstance(models, list):
            return models
    raise ValueError("Unsupported model_keys.json structure: expected list or dict with 'models'.")


def get_provider_key(keys: Any, provider: str) -> Optional[str]:
    """
    Return provider API key if present in providers map; else None.
    """
    if isinstance(keys, dict):
        return keys.get("providers", {}).get(provider, {}).get("api_key")
    return None


def find_model_by_order(models: List[Dict[str, Any]], order: int) -> Optional[Dict[str, Any]]:
    for m in models:
        if m.get("order") == order:
            return m
    return None


def find_model_by_name(models: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    # First try exact match on 'name'
    for m in models:
        if m.get("name") == name:
            return m
    # Fallback: try match on 'model_id'
    for m in models:
        if m.get("model_id") == name:
            return m
    return None


if __name__ == "__main__":
    # Minimal smoke test for manual runs
    keys = load_model_keys()
    models = get_models(keys)
    print(f"Loaded {len(models)} models from config/model_keys.json")
    # Show first model summary (redact secrets)
    if models:
        first = models[0].copy()
        if "api_key_alias" in first:
            # do not print the full secret; show only prefix
            val = str(first["api_key_alias"])
            first["api_key_alias"] = val[:6] + "..." if val else None
        print("Example model:", json.dumps({k: first.get(k) for k in ["order", "name", "model_id", "api_key_alias"]}, ensure_ascii=False))