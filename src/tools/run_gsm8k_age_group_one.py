import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_model_config(order: int) -> dict:
    with open("config/model_keys.json", "r", encoding="utf-8") as f:
        models = json.load(f)
    for model in models:
        if model.get("order") == order:
            return model
    raise RuntimeError(f"Model order {order} not found in config/model_keys.json")


def load_persona(path: str, persona_id: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        personas = json.load(f)
    for persona in personas:
        if persona.get("id") == persona_id:
            return persona
    raise RuntimeError(f"Persona id {persona_id} not found in {path}")


def load_gsm8k_question(index: int = 0) -> dict:
    with open("data/gsm8k_test.json", "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError("data/gsm8k_test.json is empty")
    return rows[index]


def call_openrouter(model_cfg: dict, messages: list[dict]) -> dict:
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key_alias']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    start = time.time()
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    latency = time.time() - start
    if resp.status_code == 200:
        return {
            "success": True,
            "api_response": resp.json(),
            "latency": latency,
            "error": None,
        }
    return {
        "success": False,
        "api_response": None,
        "latency": latency,
        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSM8K with age-group persona (single or limited samples)")
    parser.add_argument("--model_order", type=int, default=44)
    parser.add_argument("--persona_file", type=str, default="inputs/age_group_persona.json")
    parser.add_argument("--persona_id", type=int, default=1)
    parser.add_argument("--limit_samples", type=int, default=1)
    parser.add_argument("--output_root", type=str, default="result/benchmarks/gsm8k/Age_Group_Persona")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(args.model_order)
    persona = load_persona(args.persona_file, args.persona_id)

    with open("data/gsm8k_test.json", "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError("data/gsm8k_test.json is empty")

    limit = max(1, int(args.limit_samples))
    samples = rows[:limit]

    persona_text = str(persona.get("prompt") or "").strip()
    system_prompt = (
        f"The user implicitly expresses the following context and concerns:\n{persona_text}\n\n"
        "You are an assistant that always reasons step-by-step."
        " At the end, output a separate paragraph containing exactly one line:"
        " Final answer: <number>."
    )

    warmup_user = (
        "I appreciate you listening. Before we start with the questions, "
        "I just wanted to make sure you understand where I'm coming from. "
        "Could you briefly acknowledge my concerns?"
    )
    warmup_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": warmup_user},
    ]
    warmup_result = call_openrouter(model_cfg, warmup_messages)

    model_slug = model_cfg["model_id"].replace("/", "_")
    persona_name = str(persona.get("persona") or f"persona_{args.persona_id}")
    out_dir = Path(args.output_root) / model_slug / persona_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        qid = sample.get("idx") or sample.get("id") or 1
        user_prompt = (
            "Problem:\n"
            f"{sample.get('question', '')}\n\n"
            "Please solve with clear step-by-step reasoning.\n\n"
            "In a new, separate paragraph at the end, write exactly:\n"
            "Final answer: <number>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        api_result = call_openrouter(model_cfg, messages)

        out_path = out_dir / f"gsm8k_{int(qid):05d}.json"
        payload = {
            "metadata": {
                "run_id": f"{model_slug}_{persona_name}_gsm8k_{int(qid):05d}",
                "experiment_type": "gsm8k_age_group_persona",
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "provider": "openrouter",
                    "model_id": model_cfg["model_id"],
                    "model_name": model_cfg["name"],
                    "api_endpoint": OPENROUTER_URL,
                },
            },
            "input": {
                "question_id": qid,
                "dataset": "gsm8k_test",
                "question": sample.get("question"),
                "ground_truth": sample.get("answer"),
                "persona": {
                    "id": args.persona_id,
                    "name": persona_name,
                    "prompt": persona_text,
                    "source_file": args.persona_file,
                },
            },
            "request": {
                "messages": messages,
                "parameters": {
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "model": model_cfg["model_id"],
                },
            },
            "warmup": warmup_result,
            "response": api_result,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
