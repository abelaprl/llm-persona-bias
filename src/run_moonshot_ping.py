import argparse
import json
import os
from moonshot_client import load_model_config, chat, extract_text


def main():
    parser = argparse.ArgumentParser(description="Ping Moonshot/Kimi chat API")
    parser.add_argument("--model-name", default="Moonshot Kimi 8K", help="Model name in config/model_keys.json")
    parser.add_argument("--config", default="config/model_keys.json", help="Path to model config JSON")
    parser.add_argument("--prompt", default="Hello, Kimi! Say 'pong' if you can read this.", help="User prompt")
    args = parser.parse_args()

    # argparse converts dashes to underscores
    model_name = getattr(args, "model_name", None) or getattr(args, "model-name", None) or "Moonshot Kimi 8K"
    cfg = load_model_config(args.config, model_name)
    api_key = cfg.get("api_key_alias")
    endpoint = cfg.get("endpoint_url")
    model_id = cfg.get("model_id")

    if not api_key or not endpoint or not model_id:
        raise RuntimeError("Missing api_key_alias, endpoint_url, or model_id in model config entry.")

    messages = [
        {"role": "system", "content": "You are Kimi, helpful and concise."},
        {"role": "user", "content": args.prompt},
    ]
    resp = chat(messages, api_key=api_key, endpoint_url=endpoint, model_id=model_id)
    text = extract_text(resp)
    print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
