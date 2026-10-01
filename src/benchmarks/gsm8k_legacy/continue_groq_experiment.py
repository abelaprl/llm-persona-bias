#!/usr/bin/env python3
"""
Script untuk melanjutkan eksperimen Groq Llama dari ID 31 sampai 200
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

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_woman_promts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
# Default to the _v2 folder for improved structure
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/woman_explicits"

def load_data():
    """Load input prompts and GSM8K data"""
    with open(INPUT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    return prompts, gsm8k_data

def call_groq_api(messages, temperature=0.2, max_tokens=500, max_retries=3):
    """Call Groq API with retry mechanism for rate limits"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                return response.json(), latency, None
            elif response.status_code == 429:
                # Rate limit hit - wait progressively longer
                wait_time = (attempt + 1) * 45  # 45, 90, 135 seconds
                print(f"    ⏳ Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15  # 15, 30 seconds
                print(f"    ⚠️ Network error, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            return None, 0, str(e)
    
    return None, 0, "Max retries exceeded"

def create_messages(persona_prompt, gsm8k_question):
    """Create messages for the chat completion"""
    return [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": f"Solve this math problem step by step:\\n\\n{gsm8k_question}"}
    ]


def create_structured_output(gsm8k_item, persona_id, persona_prompt, gsm8k_id, api_result, latency, error):
    """Create the structured JSON format matching groq_improved_structure.py"""
    from datetime import datetime

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
            "run_id": f"groq_woman_explicit_{persona_id:03d}_{gsm8k_id:05d}",
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
                "type": "woman_explicit",
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
                "retry_count": 0
            }
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
            "reasoning_quality": None,
            "bias_indicators": None
        }
    }

    return structured

def process_single_case(persona_id, persona_prompt, gsm8k_id, gsm8k_item):
    """Process a single test case"""
    print(f"🧮 Processing GSM8K {gsm8k_id}...")

    messages = create_messages(persona_prompt, gsm8k_item["question"])

    # Call API
    result, latency, error = call_groq_api(messages)

    # Build structured output
    structured = create_structured_output(gsm8k_item, persona_id, persona_prompt, gsm8k_id, result, latency, error)

    if structured["response"]["success"]:
        print(f"  ✅ Success! Latency: {latency:.2f}s")
    else:
        print(f"  ❌ Error: {error or 'No valid response'}")

    return structured

def save_result(result, gsm8k_id):
    """Save individual result to file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = f"gsm8k_{gsm8k_id:05d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def check_existing_files(start_id, end_id):
    """Check which files already exist"""
    existing = []
    missing = []
    
    for gsm8k_id in range(start_id, end_id + 1):
        filename = f"gsm8k_{gsm8k_id:05d}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath):
            existing.append(gsm8k_id)
        else:
            missing.append(gsm8k_id)
    
    return existing, missing

def continue_experiment(start_id=31, end_id=200, output_dir=None):
    """Continue the experiment from where it left off"""
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    print(f"🔄 CONTINUING GROQ LLAMA EXPERIMENT")
    print(f"📊 Model: {MODEL_ID}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📈 Range: GSM8K ID {start_id} to {end_id}")
    print("=" * 60)
    
    # Check existing files
    existing, missing = check_existing_files(start_id, end_id)
    
    print(f"✅ Already completed: {len(existing)} files")
    print(f"⏳ Still needed: {len(missing)} files")
    
    if not missing:
        print("🎉 All files already exist! Nothing to do.")
        return
    
    print(f"📝 Will process IDs: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print()
    
    # Load data
    prompts, gsm8k_data = load_data()
    
    # Use persona ID 1
    target_persona = next((p for p in prompts if p["id"] == 1), None)
    if not target_persona:
        print("❌ Persona ID 1 not found!")
        return
    
    print(f"👤 Using Persona {target_persona['id']}: {target_persona['prompt']}")
    print()
    
    # Process missing files
    success_count = len(existing)  # Count existing as successes
    error_count = 0
    
    for i, gsm8k_id in enumerate(missing, 1):
        if gsm8k_id > len(gsm8k_data):
            print(f"⚠️ GSM8K ID {gsm8k_id} exceeds available data ({len(gsm8k_data)} items)")
            break
        
        gsm8k_item = gsm8k_data[gsm8k_id - 1]  # Convert to 0-based index
        
        # Process this case
        result = process_single_case(
            target_persona["id"],
            target_persona["prompt"],
            gsm8k_id,
            gsm8k_item
        )
        
        # Save result
        save_result(result, gsm8k_id)

        # Update counters (structured response)
        if result.get("response", {}).get("success"):
            success_count += 1
        else:
            error_count += 1
        
        # Progress update
        if i % 10 == 0 or i == len(missing):
            progress = i / len(missing) * 100
            total_processed = success_count + error_count
            print(f"📊 Progress: {progress:.1f}% ({i}/{len(missing)}) | Total Success: {success_count}/{total_processed}")
        
        # Longer delay to avoid rate limiting (Groq: 6000 TPM)
        time.sleep(3.5)  # ~17 requests per minute, well under limit
    
    # Final summary
    total_files = len(existing) + len(missing)
    final_success_rate = (success_count / total_files * 100) if total_files > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT CONTINUATION COMPLETED!")
    print(f"📊 Total Files Expected: {total_files}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Overall Success Rate: {final_success_rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue Groq experiment with flexible range and output dir")
    parser.add_argument("--start", type=int, default=31, help="Start GSM8K ID (inclusive)")
    parser.add_argument("--end", type=int, default=200, help="End GSM8K ID (inclusive)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Test API connection first
    print("🔧 Testing Groq API connection...")
    test_messages = [{"role": "user", "content": "Hello, just respond with 'OK'"}]
    result, latency, error = call_groq_api(test_messages)

    if error:
        print(f"❌ API test failed: {error}")
        sys.exit(1)
    else:
        print(f"✅ API test successful! Latency: {latency:.2f}s")
        print()

    # Continue experiment with provided args
    continue_experiment(start_id=args.start, end_id=args.end, output_dir=args.output_dir)