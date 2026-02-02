#!/usr/bin/env python3
"""
Lanjutan woman explicits ID 201-400 dengan Groq Llama
Menggunakan struktur JSON yang sudah diperbaiki
"""

import json
import os
import time
import requests
from datetime import datetime

# Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"
DELAY_SECONDS = 5  # 5 detik delay
MAX_RETRIES = 3

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_woman_promts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/woman_explicits"

def call_groq_api(messages, temperature=0.2, max_tokens=512):
    """Call Groq API with retry mechanism"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(MAX_RETRIES):
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
                    "attempt": attempt + 1
                }
            elif response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"    Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "content": None,
                    "raw_api_response": None,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "latency_sec": latency,
                    "tokens": {},
                    "attempt": attempt + 1
                }
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 5
                print(f"    Exception: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    "success": False,
                    "content": None,
                    "raw_api_response": None,
                    "error": f"Exception after {MAX_RETRIES} attempts: {str(e)}",
                    "latency_sec": 0,
                    "tokens": {},
                    "attempt": attempt + 1
                }
    
    return {
        "success": False,
        "content": None,
        "raw_api_response": None,
        "error": f"Failed after {MAX_RETRIES} attempts",
        "latency_sec": 0,
        "tokens": {},
        "attempt": MAX_RETRIES
    }

def create_structured_output(question_data, persona_data, question_id, api_response):
    """Create structured JSON output"""
    return {
        "metadata": {
            "run_id": f"groq_woman_explicit_{persona_data['id']:03d}_{question_id:05d}",
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
                "type": "woman_explicit",
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
                "retry_attempts": api_response.get("attempt", 1)
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
    """Main function untuk woman explicits ID 201-400"""
    print("🚀 GROQ LLAMA - WOMAN EXPLICITS ID 201-400")
    print("=" * 60)
    print(f"⏳ Delay: {DELAY_SECONDS}s between requests")
    print(f"🔄 Max retries: {MAX_RETRIES} attempts per request")
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
    
    # Use first prompt (ID 1) - woman explicit
    persona_data = prompts[0]
    print(f"🎯 Using persona ID {persona_data['id']}: {persona_data['prompt'][:80]}...")
    print()
    
    # Statistics
    success_count = 0
    error_count = 0
    start_time = datetime.now()
    
    # Process questions 201-400
    for i in range(200, 400):  # 200-399 -> questions 201-400
        question_id = i + 1
        output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{question_id:05d}.json")
        
        # Skip if already exists
        if os.path.exists(output_file):
            print(f"[{question_id:3d}] ⏭️  Already exists, skipping...")
            continue
        
        print(f"[{question_id:3d}] 🔄 Processing GSM8K question {question_id}...")
        
        # Check bounds
        if i >= len(gsm8k_data):
            print(f"[{question_id:3d}] ❌ Question index out of range! (max: {len(gsm8k_data)})")
            break
        
        question_data = gsm8k_data[i]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API with retries
        api_response = call_groq_api(messages)
        
        # Create structured output
        structured_data = create_structured_output(question_data, persona_data, question_id, api_response)
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
            if api_response["success"]:
                print(f"[{question_id:3d}] ✅ Success (latency: {api_response['latency_sec']:.2f}s, attempt: {api_response.get('attempt', 1)})")
                print(f"[{question_id:3d}]    Response: {api_response['content'][:60]}...")
                success_count += 1
            else:
                print(f"[{question_id:3d}] ❌ Failed: {api_response['error']}")
                error_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 20 questions
        if question_id % 20 == 0:
            total = success_count + error_count
            rate = success_count/total*100 if total > 0 else 0
            elapsed = datetime.now() - start_time
            questions_done = question_id - 200  # Since we start from 201
            avg_time_per_q = elapsed.total_seconds() / questions_done if questions_done > 0 else 0
            remaining_q = 400 - question_id
            eta_seconds = remaining_q * avg_time_per_q
            eta_minutes = eta_seconds / 60
            
            print(f"[{question_id:3d}] 📊 Progress: {questions_done}/200 | Success: {success_count} | Errors: {error_count} | Rate: {rate:.1f}%")
            print(f"[{question_id:3d}] ⏰ Elapsed: {elapsed} | ETA: {eta_minutes:.1f} minutes")
        
        # Rate limiting
        if question_id < 400:
            print(f"[{question_id:3d}] ⏳ Waiting {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)
    
    # Final summary
    total = success_count + error_count
    rate = success_count/total*100 if total > 0 else 0
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 WOMAN EXPLICITS 201-400 COMPLETED!")
    print(f"📊 Total processed: {total}/200")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success rate: {rate:.1f}%")
    print(f"⏰ Total time: {elapsed}")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()