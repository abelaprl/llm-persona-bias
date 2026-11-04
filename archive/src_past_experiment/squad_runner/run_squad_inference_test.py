"""
SQuAD v2 Inference Runner: Single sample test using OpenRouter API.

- Loads sampled SQuAD v2 dataset from CSV
- Selects 1 sample for testing
- Runs inference using allenai/olmo-3.1-32b-instruct
- Saves result in structured JSON format
- Follows TruthfulQA runner pattern

Usage:
python src/squad_runner/run_squad_inference_test.py
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful AI assistant."

SQUAD_PROMPT_TEMPLATE = """Answer the question by extracting an exact span from the context.

Rules:
- The answer must be copied exactly from the context
- Do not rephrase or paraphrase
- Use the shortest span that answers the question
- If the question is not answerable from the context, output: unanswerable

Context:
{context}

Question:
{question}

Answer:"""


def load_model_config(order: int = 42) -> dict:
    """Load model configuration from config/model_keys.json"""
    with open("config/model_keys.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    for model in models:
        if model.get("order") == order:
            return model

    raise RuntimeError(f"Model order {order} not found in config/model_keys.json")


def load_squad_sample(csv_path: str, sample_index: int = 0) -> Optional[dict]:
    """Load single sample from SQuAD v2 CSV"""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        row = df.iloc[sample_index]
        return {
            'id': row.get('id', f'sample_{sample_index}'),
            'question': row.get('question', ''),
            'context': row.get('context', ''),
            'is_impossible': bool(row.get('is_impossible', False)),
            'answers': row.get('answers', '[]'),
            'title': row.get('title', ''),
        }
    except Exception as e:
        print(f"[ERROR] Failed to load sample: {e}")
        return None


def build_squad_messages(context: str, question: str) -> list[dict]:
    """Build message array for SQuAD question"""
    user_prompt = SQUAD_PROMPT_TEMPLATE.format(context=context, question=question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_openrouter_api(
    model_cfg: dict,
    messages: list[dict],
    retry_count: int = 0,
    max_retries: int = 3
) -> dict:
    """Call OpenRouter API with retry mechanism"""
    headers = {
        "Authorization": f"Bearer {model_cfg['api_key_alias']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_cfg["model_id"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }

    start = time.time()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start

        if resp.status_code == 200:
            return {
                "success": True,
                "api_response": resp.json(),
                "latency": latency,
                "error": None,
                "retry_count": retry_count,
            }

        if resp.status_code in [429, 500, 502, 503, 504] and retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"      [RETRY] Status {resp.status_code} - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)

        return {
            "success": False,
            "api_response": None,
            "latency": latency,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "retry_count": retry_count,
        }
    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            wait_time = 5 * (2 ** retry_count)
            print(f"      [RETRY] Timeout - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": "Timeout after retries",
            "retry_count": retry_count,
        }
    except Exception as exc:
        if retry_count < max_retries:
            wait_time = 3 * (2 ** retry_count)
            print(f"      [RETRY] Exception - waiting {wait_time}s...")
            time.sleep(wait_time)
            return call_openrouter_api(model_cfg, messages, retry_count + 1, max_retries)
        return {
            "success": False,
            "api_response": None,
            "latency": time.time() - start,
            "error": str(exc),
            "retry_count": retry_count,
        }


def create_result_json(
    model_cfg: dict,
    sample: dict,
    messages: list[dict],
    api_result: dict
) -> dict:
    """Create structured result JSON"""
    timestamp = datetime.now().isoformat()

    result = {
        "metadata": {
            "run_id": f"squad_v2_{sample['id']}",
            "experiment_type": "squad_v2_inference_test",
            "dataset": "SQuAD_v2",
            "timestamp": timestamp,
            "model_info": {
                "provider": "openrouter",
                "model_id": model_cfg["model_id"],
                "model_name": model_cfg["name"],
            },
        },
        "input": {
            "sample_id": sample['id'],
            "question": sample['question'],
            "context": sample['context'],
            "title": sample.get('title', ''),
            "is_impossible": sample['is_impossible'],
            "answers_raw": sample.get('answers', ''),
        },
        "request": {
            "messages": messages,
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 512,
                "model": model_cfg["model_id"],
            },
        },
    }

    if api_result["success"]:
        api_resp = api_result["api_response"]
        choice = api_resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        answer = (message.get("content") or "").strip()
        usage = api_resp.get("usage", {})

        result["response"] = {
            "success": True,
            "answer": answer,
            "raw_api_response": api_resp,
            "error": None,
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": usage,
                "retry_count": api_result["retry_count"],
            },
        }

        result["analysis"] = {
            "answer_length": len(answer.split()) if answer else 0,
            "answer_empty": len(answer.strip()) == 0,
            "is_unanswerable_response": "unanswerable" in answer.lower(),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_sec": api_result["latency"],
        }
    else:
        result["response"] = {
            "success": False,
            "answer": None,
            "raw_api_response": None,
            "error": api_result["error"],
            "performance": {
                "latency_sec": api_result["latency"],
                "tokens": {},
                "retry_count": api_result["retry_count"],
            },
        }

        result["analysis"] = {
            "answer_length": 0,
            "answer_empty": True,
            "is_unanswerable_response": False,
            "total_tokens": 0,
            "latency_sec": api_result["latency"],
        }

    return result


def save_result_json(result_json: dict, output_dir: Path) -> Path:
    """Save result to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = result_json["input"]["sample_id"]
    filepath = output_dir / f"{sample_id}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    return filepath


def csv_row_from_result(result_json: dict) -> dict:
    """Convert result JSON to CSV row"""
    inp = result_json.get("input", {})
    analysis = result_json.get("analysis", {})
    response = result_json.get("response", {})
    
    return {
        "id": inp.get("sample_id", ""),
        "question": inp.get("question", ""),
        "context_length": len(inp.get("context", "").split()),
        "is_impossible": inp.get("is_impossible", False),
        "model_answer": response.get("answer", "") or "",
        "answer_length": analysis.get("answer_length", 0),
        "answer_empty": analysis.get("answer_empty", True),
        "is_unanswerable_response": analysis.get("is_unanswerable_response", False),
        "total_tokens": analysis.get("total_tokens", 0),
        "latency_sec": analysis.get("latency_sec", 0.0),
        "status": "SUCCESS" if response.get("success") else "FAILED",
    }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="SQuAD v2 Inference Test Runner")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"data\squad_v2_sampled_10percent.csv",
        help="Path to SQuAD v2 sampled CSV"
    )
    parser.add_argument(
        "--model_order",
        type=int,
        default=42,
        help="Model order from config/model_keys.json"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index to test (0-based)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SQuAD v2 INFERENCE TEST RUNNER")
    print("=" * 80)

    # Load model config
    try:
        model_cfg = load_model_config(args.model_order)
        print(f"✓ Model: {model_cfg['name']} ({model_cfg['model_id']})")
    except RuntimeError as e:
        print(f"✗ {e}")
        return

    # Load sample
    print(f"✓ Loading sample from: {args.input_csv}")
    sample = load_squad_sample(args.input_csv, args.sample_index)
    if not sample:
        print("✗ Failed to load sample")
        return

    print(f"\n{'='*80}")
    print("SAMPLE DETAILS")
    print(f"{'='*80}")
    print(f"ID: {sample['id']}")
    print(f"Is Impossible: {sample['is_impossible']}")
    print(f"Question: {sample['question']}")
    print(f"Context: {sample['context'][:200]}..." if len(sample['context']) > 200 else f"Context: {sample['context']}")
    print(f"{'='*80}\n")

    # Build messages
    print("→ Building prompt...")
    messages = build_squad_messages(sample['context'], sample['question'])
    print(f"  Messages built: {len(messages)} items")

    # Call API
    print("→ Calling OpenRouter API...")
    api_result = call_openrouter_api(model_cfg, messages)

    if api_result["success"]:
        print(f"  ✓ API call succeeded")
        answer = api_result["api_response"]["choices"][0]["message"]["content"]
        print(f"\n{'='*80}")
        print("MODEL RESPONSE")
        print(f"{'='*80}")
        print(f"Answer: {answer}")
        print(f"{'='*80}\n")
    else:
        print(f"  ✗ API call failed: {api_result['error']}")

    # Create result JSON
    result_json = create_result_json(model_cfg, sample, messages, api_result)

    # Save result
    model_slug = model_cfg["model_id"].replace("/", "_")
    output_dir = Path("result") / "TRUTHFULQA" / model_slug
    output_path = save_result_json(result_json, output_dir)
    print(f"✓ Result saved: {output_path}")

    # Save CSV
    csv_path = output_dir / "test_single_sample.csv"
    csv_row = csv_row_from_result(result_json)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'id', 'question', 'context_length', 'is_impossible',
            'model_answer', 'answer_length', 'answer_empty',
            'is_unanswerable_response', 'total_tokens', 'latency_sec', 'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_row)
    
    print(f"✓ CSV saved: {csv_path}")

    # Final summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Status: {'✓ SUCCESS' if result_json['response']['success'] else '✗ FAILED'}")
    print(f"Sample ID: {sample['id']}")
    print(f"Model: {model_cfg['name']}")
    print(f"Latency: {result_json['analysis']['latency_sec']:.2f}s")
    print(f"Answer Length: {result_json['analysis']['answer_length']} words")
    print(f"Answer Empty: {result_json['analysis']['answer_empty']}")
    print(f"Total Tokens: {result_json['analysis']['total_tokens']}")
    print(f"Output JSON: {output_path}")
    print(f"Output CSV: {csv_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
