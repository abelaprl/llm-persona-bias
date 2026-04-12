import argparse
import os
import json
import time
import copy
import pandas as pd
import requests
from datetime import datetime

def load_api_key(model_keys_path, model_id):
    with open(model_keys_path, encoding="utf-8") as f:
        keys = json.load(f)
    for entry in keys:
        if entry.get("model_id") == model_id:
            return entry["api_key_alias"]
    raise ValueError(f"API key for model_id {model_id} not found in {model_keys_path}")

def build_run_id(model_slug, persona_type, persona_id, qid):
    return f"{model_slug}_{persona_type}_{persona_id}_gsm8k_{qid:05d}"

def build_messages(persona, question):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    if persona["type"].endswith("_implicit"):
        messages.append({"role": "user", "content": persona["warmup_prompt"]})
        messages.append({"role": "assistant", "content": persona["warmup_response"]})
        messages.append({"role": "user", "content": f"Solve the following math problem:\n\n{question}\n\nAnswer:"})
    elif persona["type"].endswith("_explicit"):
        messages.append({"role": "user", "content": f"{persona['explicit_prompt']} {question}\n\nAnswer:"})
    else:
        raise ValueError(f"Unknown persona type: {persona['type']}")
    return messages

def call_openrouter_api(api_key, model_id, messages, max_tokens, temperature):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload)
    latency = time.time() - t0
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = {"error": resp.text}
    return resp_json, latency

def update_json(template, persona, question, max_tokens, api_key, model_id, temperature):
    # Deepcopy to avoid mutating original
    data = copy.deepcopy(template)
    # Update metadata
    now_iso = datetime.now().isoformat()
    qid = data["input"]["question_id"]
    persona_type = persona["type"]
    persona_id = persona["id"]
    model_slug = model_id.replace("/", "_")
    data["metadata"]["run_id"] = build_run_id(model_slug, persona_type, persona_id, qid)
    data["metadata"]["timestamp"] = now_iso
    data["metadata"]["model_info"]["provider"] = "openrouter"
    data["metadata"]["model_info"]["model_id"] = model_id
    # model_name: keep as in template
    data["metadata"]["model_info"]["api_endpoint"] = "https://openrouter.ai/api/v1/chat/completions"
    # Update input (keep all keys, just update question if needed)
    data["input"]["question"] = question
    # Update request
    data["request"]["messages"] = build_messages(persona, question)
    data["request"]["parameters"]["max_tokens"] = max_tokens
    data["request"]["parameters"]["model"] = model_id
    # temperature: keep as in template or set to 0.2 if missing
    if "temperature" in data["request"]["parameters"]:
        data["request"]["parameters"]["temperature"] = data["request"]["parameters"].get("temperature", 0.2)
    else:
        data["request"]["parameters"]["temperature"] = 0.2
    # Call API
    api_messages = data["request"]["messages"]
    api_temp = data["request"]["parameters"]["temperature"]
    resp_json, latency = call_openrouter_api(api_key, model_id, api_messages, max_tokens, api_temp)
    # Update response
    content = ""
    success = False
    if resp_json and "choices" in resp_json and resp_json["choices"]:
        content = resp_json["choices"][0]["message"].get("content", "")
        success = True
    data["response"]["success"] = success
    data["response"]["content"] = content
    data["response"]["raw_api_response"] = resp_json
    data["response"]["latency"] = latency
    # If template has error/performance, update/clear as needed
    if "error" in data["response"]:
        data["response"]["error"] = None if success else resp_json.get("error", "API error")
    if "performance" in data["response"]:
        data["response"]["performance"]["latency_sec"] = latency
        usage = resp_json.get("usage", {})
        data["response"]["performance"]["tokens"] = usage
        data["response"]["performance"]["retry_count"] = 0
    # Reset analysis fields to null (preserve keys)
    if "analysis" in data:
        for k in data["analysis"]:
            data["analysis"][k] = None
    # retry_count
    data["retry_count"] = 0
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_json_path", required=True)
    parser.add_argument("--input_examples_csv", required=True)
    parser.add_argument("--json_root_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_keys", required=True)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    # Load template for schema
    with open(args.template_json_path, encoding="utf-8") as f:
        template = json.load(f)
    # Load input CSV
    df = pd.read_csv(args.input_examples_csv)
    # Get model_id from template
    model_id = template["metadata"]["model_info"]["model_id"]
    api_key = load_api_key(args.model_keys, model_id)
    # For each row in CSV
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        # Flexible: support persona_type/persona_id or persona
        if "persona_type" in row and "persona_id" in row:
            persona_type = row["persona_type"]
            persona_id = int(row["persona_id"])
        elif "persona" in row and isinstance(row["persona"], str) and "_" in row["persona"]:
            # Example: woman_implicit_3
            persona_parts = row["persona"].rsplit("_", 1)
            persona_type = persona_parts[0]
            persona_id = int(persona_parts[1])
        else:
            print(f"Could not parse persona_type/persona_id for row: {row}")
            continue
        persona_dir = f"{persona_type}_{persona_id}"
        json_path = os.path.join(args.json_root_dir, persona_dir, f"gsm8k_{qid:05d}.json")
        if not os.path.exists(json_path):
            print(f"Template JSON not found: {json_path}")
            continue
        with open(json_path, encoding="utf-8") as f:
            template_sample = json.load(f)
        # Get question and persona from template
        question = template_sample["input"]["question"]
        persona = template_sample["input"]["persona"]
        # Update JSON
        new_json = update_json(template_sample, persona, question, args.max_tokens, api_key, model_id, template_sample["request"]["parameters"].get("temperature", 0.2))
        # Output path
        out_persona_dir = os.path.join(args.out_dir, persona_dir)
        os.makedirs(out_persona_dir, exist_ok=True)
        out_path = os.path.join(out_persona_dir, f"gsm8k_{qid:05d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_json, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
