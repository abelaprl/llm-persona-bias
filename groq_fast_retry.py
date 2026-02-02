#!/usr/bin/env python3
"""
Fast Groq experiment with retry mechanism
Reduced delay + automatic retry on failures
"""

import json
import os
import time
import requests
from datetime import datetime
import random

# Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"

# Faster timing settings
NORMAL_DELAY = 5      # 5 detik normal delay
ERROR_DELAY = 10      # 10 detik kalau ada error
MAX_RETRIES = 3       # Max 3x retry per question

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_man_prompts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/man_explicits_v2"

def call_groq_api_with_retry(messages, max_retries=MAX_RETRIES):
    """Call Groq API with retry mechanism"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512
    }
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "content": result["choices"][0]["message"]["content"],
                    "raw_api_response": result,
                    "error": None,
                    "latency_sec": latency,
                    "tokens": result.get("usage", {}),
                    "attempts": attempt + 1
                }
            elif response.status_code == 429:  # Rate limit
                error_msg = f"Rate limit (attempt {attempt + 1}/{max_retries})"
                print(f"    ⏳ {error_msg}, waiting {ERROR_DELAY}s...")
                if attempt < max_retries - 1:
                    time.sleep(ERROR_DELAY)
                    continue
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < max_retries - 1:
                    print(f"    🔄 Error (attempt {attempt + 1}): {error_msg}, retrying...")
                    time.sleep(5)
                    continue
            
            # Final attempt failed
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": error_msg,
                "latency_sec": latency if 'latency' in locals() else 0,
                "tokens": {},
                "attempts": attempt + 1
            }
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            if attempt < max_retries - 1:
                print(f"    💥 Exception (attempt {attempt + 1}): {error_msg}, retrying...")
                time.sleep(5)
                continue
            
            # Final attempt failed
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": error_msg,
                "latency_sec": 0,
                "tokens": {},
                "attempts": attempt + 1
            }

def create_structured_output(question_data, persona_data, question_id, api_response):
    """Create structured JSON output (same as before)"""
    return {
        "metadata": {
            "run_id": f"groq_man_explicit_{persona_data['id']:03d}_{question_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "groq",
                "model_id": MODEL_ID,
                "api_endpoint": GROQ_API_URL
            }
        },
        "input": {
            "question_id": question_id,
            "dataset": "gsm8k_test",
            "question": question_data["question"],
            "ground_truth": question_data["answer"],
            "persona": {
                "id": persona_data["id"],
                "type": "man_explicit",
                "prompt": persona_data["prompt"]
            }
        },
        "request": {
            "messages": [
                {"role": "system", "content": persona_data["prompt"]},
                {"role": "user", "content": question_data["question"]}
            ],
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 512,
                "model": MODEL_ID
            }
        },
        "response": {
            "success": api_response["success"],
            "content": api_response["content"],
            "raw_api_response": api_response["raw_api_response"],
            "error": api_response["error"],
            "performance": {
                "latency_sec": api_response["latency_sec"],
                "tokens": api_response["tokens"],
                "attempts": api_response.get("attempts", 1)
            }
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
            "reasoning_quality": None,
            "bias_indicators": None
        }
    }

def main():
    """Main function with faster execution and retry"""
    print("🚀 GROQ LLAMA EXPERIMENT - FAST MODE WITH RETRY")
    print("=" * 60)
    print(f"⚡ Normal delay: {NORMAL_DELAY}s | Error delay: {ERROR_DELAY}s")
    print(f"🔄 Max retries per question: {MAX_RETRIES}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    with open(INPUT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    print(f"✅ Loaded {len(prompts)} prompts and {len(gsm8k_data)} GSM8K questions")
    
    # Use first prompt (ID 1)
    persona_data = prompts[0]
    print(f"🎯 Using persona ID {persona_data['id']}: {persona_data['prompt'][:50]}...")
    print()
    
    # Statistics
    success_count = 0
    error_count = 0
    retry_count = 0
    
    # Process questions 1-200
    for i in range(200):
        question_id = i + 1
        output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{question_id:05d}.json")
        
        # Skip if already exists and is valid
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if existing_data.get("response", {}).get("success", False):
                        print(f"[{question_id:3d}] ✅ Already completed successfully, skipping...")
                        success_count += 1
                        continue
                    else:
                        print(f"[{question_id:3d}] ❌ Existing file failed, retrying...")
            except:
                print(f"[{question_id:3d}] 🔧 Invalid file, retrying...")
        
        print(f"[{question_id:3d}] 🔄 Processing GSM8K question {question_id}...")
        
        # Check bounds
        if i >= len(gsm8k_data):
            print(f"[{question_id:3d}] ❌ Question index out of range!")
            break
        
        question_data = gsm8k_data[i]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API with retry
        api_response = call_groq_api_with_retry(messages)
        
        # Track retries
        if api_response.get("attempts", 1) > 1:
            retry_count += 1
        
        # Create structured output
        structured_data = create_structured_output(question_data, persona_data, question_id, api_response)
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
            if api_response["success"]:
                print(f"[{question_id:3d}] ✅ Success (latency: {api_response['latency_sec']:.2f}s, attempts: {api_response.get('attempts', 1)})")
                success_count += 1
            else:
                print(f"[{question_id:3d}] ❌ Failed after {api_response.get('attempts', 1)} attempts: {api_response['error']}")
                error_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 10 questions
        if question_id % 10 == 0:
            total = success_count + error_count
            rate = success_count/total*100 if total > 0 else 0
            print(f"[{question_id:3d}] 📊 Progress: {question_id}/200 | Success: {success_count} | Errors: {error_count} | Retries: {retry_count} | Rate: {rate:.1f}%")
        
        # Smart delay based on result
        if question_id < 200:
            if api_response["success"]:
                delay = NORMAL_DELAY + random.uniform(-1, 1)  # Add some jitter
                print(f"[{question_id:3d}] ⏳ Waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Longer wait after errors
                delay = ERROR_DELAY
                print(f"[{question_id:3d}] ⏳ Error cooldown {delay}s...")
                time.sleep(delay)
    
    # Final summary
    total = success_count + error_count
    rate = success_count/total*100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED!")
    print(f"📊 Total processed: {total}/200")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"🔄 Questions with retries: {retry_count}")
    print(f"📈 Success rate: {rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()