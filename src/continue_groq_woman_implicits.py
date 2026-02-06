#!/usr/bin/env python3
"""
Continue Groq Llama woman_implicits experiment.
Skips existing successful files, auto-retry on errors.
Structure matches existing woman_implicits files.
"""

import json
import os
import time
import requests
import argparse
import sys
from datetime import datetime

# Konfigurasi
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"

# Paths - note the typo in filename matches original
INPUT_PROMPTS_PATH = "inputs/implicits_woman_promt.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/woman_implicits"

# Retry configuration based on error type
RETRY_CONFIG = {
    "rate_limit": {"max_retries": 10, "base_wait": 60, "multiplier": 1.5},
    "server_error": {"max_retries": 5, "base_wait": 30, "multiplier": 2},
    "timeout": {"max_retries": 5, "base_wait": 15, "multiplier": 1.5},
    "network": {"max_retries": 5, "base_wait": 10, "multiplier": 2},
    "default": {"max_retries": 3, "base_wait": 15, "multiplier": 2},
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
    else:
        return "default"

def load_data():
    """Load input prompts and GSM8K data"""
    with open(INPUT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    return prompts, gsm8k_data

def call_groq_api_with_retry(messages):
    """Call Groq API with smart retry - retries until success"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512
    }
    
    total_retries = 0
    consecutive_errors = {}
    
    while True:
        try:
            start_time = time.time()
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
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
            
            consecutive_errors[error_type] = consecutive_errors.get(error_type, 0) + 1
            wait_time = min(config["base_wait"] * (config["multiplier"] ** (consecutive_errors[error_type] - 1)), 300)
            
            total_retries += 1
            print(f"      ⚠️ [{error_type.upper()}] HTTP {response.status_code}: {error_msg[:80]}")
            print(f"         Retry #{total_retries}, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            
        except requests.exceptions.Timeout:
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

def check_existing_file(gsm8k_id):
    """Check if a successful result already exists"""
    output_file = f"{OUTPUT_DIR}/gsm8k_{gsm8k_id:05d}.json"
    
    if not os.path.exists(output_file):
        return False, None
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if data.get("response", {}).get("success", False):
            return True, data
        else:
            return False, None
    except:
        return False, None

def create_structured_output(gsm8k_item, persona_id, persona_prompt, gsm8k_id, api_result, latency, error, retry_count=0):
    """Create the structured JSON format matching existing woman_implicits structure"""
    
    success = (api_result is not None and "choices" in api_result)
    content = None
    raw_api = api_result if api_result is not None else None
    tokens = api_result.get("usage", {}) if api_result else {}

    if success:
        try:
            content = api_result["choices"][0]["message"]["content"]
        except Exception:
            content = None

    structured = {
        "metadata": {
            "run_id": f"groq_woman_implicit_{persona_id:03d}_{gsm8k_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "groq",
                "model_id": MODEL_ID,
                "api_endpoint": GROQ_API_URL
            }
        },
        "input": {
            "question_id": gsm8k_id,
            "dataset": "gsm8k_test",
            "question": gsm8k_item["question"],
            "ground_truth": gsm8k_item.get("answer"),
            "persona": {
                "id": persona_id,
                "type": "woman_implicit",
                "prompt": persona_prompt
            }
        },
        "request": {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": gsm8k_item["question"]}
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 512,
                "model": MODEL_ID
            }
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

def save_result(result, gsm8k_id):
    """Save individual result to file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = f"gsm8k_{gsm8k_id:05d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def continue_experiment(start_id=1, end_id=1319):
    """Continue the experiment from where it left off"""
    print(f"🔄 CONTINUING GROQ WOMAN_IMPLICITS EXPERIMENT")
    print(f"📊 Model: {MODEL_ID}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📈 Range: GSM8K ID {start_id} to {end_id}")
    print("=" * 60)
    
    # Check existing files
    existing_count = 0
    missing_ids = []
    
    for gsm8k_id in range(start_id, end_id + 1):
        exists, _ = check_existing_file(gsm8k_id)
        if exists:
            existing_count += 1
        else:
            missing_ids.append(gsm8k_id)
    
    print(f"✅ Already completed: {existing_count} files")
    print(f"⏳ Still needed: {len(missing_ids)} files")
    
    if not missing_ids:
        print("🎉 All files already exist! Nothing to do.")
        return
    
    print(f"📝 Will process IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
    print()
    
    # Load data
    prompts, gsm8k_data = load_data()
    
    # Use persona ID 1
    target_persona = next((p for p in prompts if p["id"] == 1), None)
    if not target_persona:
        print("❌ Persona ID 1 not found!")
        return
    
    print(f"👤 Using Persona {target_persona['id']}")
    print()
    
    # Process missing files
    success_count = existing_count
    error_count = 0
    
    for i, gsm8k_id in enumerate(missing_ids, 1):
        if gsm8k_id > len(gsm8k_data):
            print(f"⚠️ GSM8K ID {gsm8k_id} exceeds available data ({len(gsm8k_data)} items)")
            break
        
        gsm8k_item = gsm8k_data[gsm8k_id - 1]
        
        print(f"🧮 [{i}/{len(missing_ids)}] Processing GSM8K {gsm8k_id}...")
        
        # Build messages - system prompt (persona) + user question
        messages = [
            {"role": "system", "content": target_persona["prompt"]},
            {"role": "user", "content": gsm8k_item["question"]}
        ]
        
        # Call API with auto-retry
        result, latency, error, retry_count = call_groq_api_with_retry(messages)
        
        # Build structured output
        structured = create_structured_output(
            gsm8k_item, target_persona["id"], target_persona["prompt"],
            gsm8k_id, result, latency, error, retry_count
        )
        
        # Save result
        save_result(structured, gsm8k_id)

        if structured["response"]["success"]:
            retry_info = f" (retries: {retry_count})" if retry_count > 0 else ""
            print(f"  ✅ Success! Latency: {latency:.2f}s{retry_info}")
            success_count += 1
        else:
            print(f"  ❌ Error: {error or 'No valid response'}")
            error_count += 1
        
        # Progress update
        if i % 50 == 0:
            progress = i / len(missing_ids) * 100
            print(f"📊 Progress: {progress:.1f}% ({i}/{len(missing_ids)}) | Total Success: {success_count}")
        
        # Rate limiting
        time.sleep(3.5)
    
    # Final summary
    total_files = end_id - start_id + 1
    final_success_rate = (success_count / total_files * 100) if total_files > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED!")
    print(f"📊 Total Files Expected: {total_files}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success Rate: {final_success_rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue Groq woman_implicits experiment")
    parser.add_argument("--start", type=int, default=1, help="Start GSM8K ID (inclusive)")
    parser.add_argument("--end", type=int, default=1319, help="End GSM8K ID (inclusive)")
    args = parser.parse_args()

    # Test API connection first
    print("🔧 Testing Groq API connection...")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    test_payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 10}
    
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=test_payload, timeout=30)
        if resp.status_code == 200:
            print(f"✅ API test successful!")
        else:
            print(f"❌ API test failed: {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ API test failed: {e}")
        sys.exit(1)
    
    print()
    continue_experiment(start_id=args.start, end_id=args.end)
