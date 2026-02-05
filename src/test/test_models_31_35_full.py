#!/usr/bin/env python3
"""
Test models 31-35 with GSM8K questions and structured JSON output.
Matches exact Groq JSON format. Handles reasoning models with different params.
Features:
- Skip existing successful files
- Auto-retry with different strategies based on error type
"""

import json
import os
import time
import requests
from datetime import datetime

# Models configuration
# - has_reasoning=True: max_tokens=2048, reasoning:{enabled:true}
# - has_reasoning=False: max_tokens=512, no reasoning param
MODELS = [
    {"order": 31, "model_id": "openai/gpt-oss-20b", "folder": "openai_gpt-oss-20b", "has_reasoning": True},
    {"order": 32, "model_id": "qwen/qwen3-next-80b-a3b-thinking", "folder": "qwen_qwen3-next-80b-a3b-thinking", "has_reasoning": True},
    {"order": 33, "model_id": "qwen/qwen3-next-80b-a3b-instruct", "folder": "qwen_qwen3-next-80b-a3b-instruct", "has_reasoning": False},
    {"order": 34, "model_id": "openai/gpt-oss-120b", "folder": "openai_gpt-oss-120b", "has_reasoning": True},
    {"order": 35, "model_id": "meta-llama/llama-3.3-70b-instruct", "folder": "meta-llama_llama-3.3-70b-instruct", "has_reasoning": False},
]

# API configuration
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Paths
PERSONA_PATH = "inputs/explicits_man_prompts.json"
GSM8K_PATH = "data/gsm8k_test.json"

# Retry configuration based on error type
RETRY_CONFIG = {
    "rate_limit": {"max_retries": 10, "base_wait": 60, "multiplier": 1.5},      # 429 - rate limit
    "server_error": {"max_retries": 5, "base_wait": 30, "multiplier": 2},       # 500, 502, 503, 504
    "timeout": {"max_retries": 5, "base_wait": 15, "multiplier": 1.5},          # Connection timeout
    "network": {"max_retries": 5, "base_wait": 10, "multiplier": 2},            # Network errors
    "model_busy": {"max_retries": 8, "base_wait": 45, "multiplier": 1.5},       # Model overloaded
    "default": {"max_retries": 3, "base_wait": 15, "multiplier": 2},            # Other errors
}

def classify_error(status_code=None, error_msg=""):
    """Classify error type for retry strategy"""
    error_msg_lower = error_msg.lower() if error_msg else ""
    
    if status_code == 429 or "rate limit" in error_msg_lower:
        return "rate_limit"
    elif status_code in [500, 502, 503, 504]:
        return "server_error"
    elif "timeout" in error_msg_lower or "timed out" in error_msg_lower:
        return "timeout"
    elif any(x in error_msg_lower for x in ["connection", "network", "dns", "refused"]):
        return "network"
    elif any(x in error_msg_lower for x in ["overloaded", "busy", "capacity", "queue"]):
        return "model_busy"
    else:
        return "default"

def call_openrouter_api_with_retry(model_id, messages, has_reasoning=False):
    """Call OpenRouter API with smart retry based on error type - retries until success"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/user/project",
        "X-Title": "Model Test",
    }
    
    # Build payload based on model type
    if has_reasoning:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
            "reasoning": {"enabled": True}
        }
    else:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512
        }
    
    total_retries = 0
    consecutive_errors = {}  # Track consecutive errors by type
    
    while True:  # Keep retrying until success
        try:
            start_time = time.time()
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if total_retries > 0:
                    print(f"      ✅ Success after {total_retries} retries!")
                return result, latency, None, total_retries
            
            # Error handling
            error_msg = ""
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text[:200])
            except:
                error_msg = response.text[:200]
            
            error_type = classify_error(response.status_code, error_msg)
            config = RETRY_CONFIG[error_type]
            
            # Track consecutive errors of this type
            consecutive_errors[error_type] = consecutive_errors.get(error_type, 0) + 1
            
            # Calculate wait time with exponential backoff
            wait_time = min(config["base_wait"] * (config["multiplier"] ** (consecutive_errors[error_type] - 1)), 300)
            
            total_retries += 1
            print(f"      ⚠️ [{error_type.upper()}] HTTP {response.status_code}: {error_msg[:80]}")
            print(f"         Retry #{total_retries}, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            
        except requests.exceptions.Timeout as e:
            error_type = "timeout"
            config = RETRY_CONFIG[error_type]
            consecutive_errors[error_type] = consecutive_errors.get(error_type, 0) + 1
            wait_time = min(config["base_wait"] * (config["multiplier"] ** (consecutive_errors[error_type] - 1)), 180)
            
            total_retries += 1
            print(f"      ⚠️ [TIMEOUT] Request timed out")
            print(f"         Retry #{total_retries}, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            
        except requests.exceptions.ConnectionError as e:
            error_type = "network"
            config = RETRY_CONFIG[error_type]
            consecutive_errors[error_type] = consecutive_errors.get(error_type, 0) + 1
            wait_time = min(config["base_wait"] * (config["multiplier"] ** (consecutive_errors[error_type] - 1)), 180)
            
            total_retries += 1
            print(f"      ⚠️ [NETWORK] Connection error: {str(e)[:60]}")
            print(f"         Retry #{total_retries}, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            
        except Exception as e:
            error_type = "default"
            config = RETRY_CONFIG[error_type]
            consecutive_errors[error_type] = consecutive_errors.get(error_type, 0) + 1
            wait_time = min(config["base_wait"] * (config["multiplier"] ** (consecutive_errors[error_type] - 1)), 120)
            
            total_retries += 1
            print(f"      ⚠️ [ERROR] {str(e)[:80]}")
            print(f"         Retry #{total_retries}, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)

def check_existing_file(output_dir, gsm8k_id):
    """Check if a successful result already exists"""
    output_file = f"{output_dir}/gsm8k_{gsm8k_id:05d}.json"
    
    if not os.path.exists(output_file):
        return False, None
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if the response was successful
        if data.get("response", {}).get("success", False):
            return True, data
        else:
            # File exists but was an error - we should retry
            return False, None
    except:
        # File is corrupted or unreadable - retry
        return False, None

def load_data():
    """Load persona and GSM8K data"""
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        personas = json.load(f)
    
    with open(GSM8K_PATH, "r", encoding="utf-8") as f:
        gsm8k = json.load(f)
    
    return personas, gsm8k

def create_structured_output(model_id, folder_name, gsm8k_id, question_text, ground_truth, 
                            persona_id, persona_prompt, api_result, latency, error, has_reasoning=False, retry_count=0):
    """Create structured JSON output matching Groq format exactly"""
    
    success = (api_result is not None and "choices" in api_result)
    content = None
    raw_api = api_result
    
    # Build tokens dict
    tokens = {}
    if api_result and "usage" in api_result:
        usage = api_result["usage"]
        tokens = {
            "queue_time": usage.get("queue_time", 0),
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_time": usage.get("prompt_time", 0),
            "completion_tokens": usage.get("completion_tokens"),
            "completion_time": usage.get("completion_time", 0),
            "total_tokens": usage.get("total_tokens"),
            "total_time": usage.get("total_time", 0)
        }
        # Add reasoning_tokens if available (from completion_tokens_details)
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details and "reasoning_tokens" in completion_details:
            tokens["reasoning_tokens"] = completion_details["reasoning_tokens"]
        else:
            tokens["reasoning_tokens"] = 0  # Default to 0 if not available
    
    if success:
        try:
            content = api_result["choices"][0]["message"].get("content")
        except Exception:
            content = None
    
    # Build parameters for request section
    if has_reasoning:
        request_params = {
            "temperature": 0.2,
            "max_tokens": 2048,
            "model": model_id,
            "reasoning": {"enabled": True}
        }
    else:
        request_params = {
            "temperature": 0.2,
            "max_tokens": 512,
            "model": model_id
        }
    
    # Create run_id matching Groq format: {folder_name}_man_explicit_{persona_id:03d}_{gsm8k_id:05d}
    run_id = f"{folder_name.replace('-', '_')}_man_explicit_{persona_id:03d}_{gsm8k_id:05d}"
    
    structured = {
        "metadata": {
            "run_id": run_id,
            "experiment_type": "gsm8k_bias_study",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_id,
                "api_endpoint": OPENROUTER_URL
            }
        },
        "input": {
            "question_id": gsm8k_id,
            "dataset": "gsm8k_test",
            "question": question_text,
            "ground_truth": ground_truth,
            "persona": {
                "id": persona_id,
                "type": "man_explicit",
                "prompt": persona_prompt
            }
        },
        "request": {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question_text}
            ],
            "parameters": request_params
        },
        "response": {
            "success": bool(success),
            "content": content,
            "raw_api_response": raw_api,
            "error": error,
            "performance": {
                "latency_sec": latency,
                "tokens": tokens,
                "retry_count": retry_count
            }
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None
        }
    }
    
    return structured

def process_model(model, personas, gsm8k_data, start_id=1, end_id=1):
    """Process a single model with range of questions"""
    model_id = model["model_id"]
    folder = model["folder"]
    has_reasoning = model["has_reasoning"]
    order = model["order"]
    
    output_dir = f"results/{folder}/man_explicit"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get persona (ID 1)
    persona = next((p for p in personas if p["id"] == 1), None)
    if not persona:
        print(f"  ❌ Persona ID 1 not found!")
        return 0, 0, 0
    
    print(f"\n[Model {order}] {model_id}")
    print(f"  📁 Output: {output_dir}")
    print(f"  🧠 Reasoning: {has_reasoning}")
    print(f"  📝 Processing questions {start_id}-{end_id}...")
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for gsm8k_id in range(start_id, end_id + 1):
        if gsm8k_id > len(gsm8k_data):
            print(f"  ⚠️ Question {gsm8k_id} exceeds available data ({len(gsm8k_data)})")
            break
        
        # Check if file already exists and is successful
        exists, existing_data = check_existing_file(output_dir, gsm8k_id)
        if exists:
            print(f"    [{gsm8k_id:5d}] ⏭️  Skipped (already exists)")
            skipped_count += 1
            success_count += 1
            continue
        
        gsm8k_item = gsm8k_data[gsm8k_id - 1]
        
        # Build messages
        messages = [
            {"role": "system", "content": persona["prompt"]},
            {"role": "user", "content": gsm8k_item["question"]}
        ]
        
        # Call API with auto-retry until success
        api_result, latency, error, retry_count = call_openrouter_api_with_retry(model_id, messages, has_reasoning)
        
        # Create structured output
        structured = create_structured_output(
            model_id, folder, gsm8k_id, gsm8k_item["question"], gsm8k_item.get("answer"),
            persona["id"], persona["prompt"], api_result, latency, error, has_reasoning, retry_count
        )
        
        # Save to file
        output_file = f"{output_dir}/gsm8k_{gsm8k_id:05d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
        
        # Report
        if structured["response"]["success"]:
            retry_info = f" (retries: {retry_count})" if retry_count > 0 else ""
            print(f"    [{gsm8k_id:5d}] ✅ Success (latency: {latency:.2f}s){retry_info}")
            success_count += 1
        else:
            print(f"    [{gsm8k_id:5d}] ❌ Error: {error or 'No response'}")
            error_count += 1
        
        # Rate limiting
        time.sleep(3)
    
    # Summary
    total = success_count + error_count
    rate = (success_count / total * 100) if total > 0 else 0
    print(f"  📊 Summary: {success_count}/{total} successful ({rate:.1f}%), {skipped_count} skipped")
    
    return success_count, error_count, skipped_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test models 31-35 with GSM8K")
    parser.add_argument("--model", type=int, default=None, help="Model order (31-35). If not specified, run all models.")
    parser.add_argument("--start", type=int, default=1, help="Start GSM8K ID (default: 1)")
    parser.add_argument("--end", type=int, default=1, help="End GSM8K ID (default: 1)")
    args = parser.parse_args()
    
    print("🧪 TESTING MODELS 31-35 WITH GSM8K QUESTIONS")
    print("=" * 80)
    print("📌 Features: Skip existing files, Auto-retry on errors")
    
    # Load data
    print("\n📂 Loading data...")
    personas, gsm8k_data = load_data()
    print(f"✅ Loaded {len(personas)} personas and {len(gsm8k_data)} GSM8K questions")
    
    total_success = 0
    total_error = 0
    total_skipped = 0
    
    # Filter models if specific model requested
    if args.model:
        models_to_run = [m for m in MODELS if m["order"] == args.model]
        if not models_to_run:
            print(f"❌ Model {args.model} not found!")
            return
    else:
        models_to_run = MODELS
    
    # Process each model
    for model in models_to_run:
        try:
            s, e, sk = process_model(model, personas, gsm8k_data, start_id=args.start, end_id=args.end)
            total_success += s
            total_error += e
            total_skipped += sk
        except Exception as ex:
            print(f"  ❌ Exception: {str(ex)}")
            total_error += 1
    
    print("\n" + "=" * 80)
    print(f"✅ TEST COMPLETED: {total_success} success, {total_error} errors, {total_skipped} skipped")

if __name__ == "__main__":
    main()