#!/usr/bin/env python3
"""
Groq experiment script - Continue from ID 201-400
Fast execution with retry mechanism
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

# Timing settings
DELAY_SECONDS = 4    # Fast 4s delay
RETRY_DELAY = 10     # 10s on rate limit
MAX_RETRIES = 3      # Max retry attempts

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_man_prompts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/man_explicits_v2"

def call_groq_api(messages, temperature=0.2, max_tokens=512, retry_count=0):
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
    
    try:
        start_time = time.time()
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
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
                "retry_count": retry_count
            }
        elif response.status_code == 429 and retry_count < MAX_RETRIES:
            # Rate limit - retry
            print(f"    ⚠️  Rate limit hit, retrying in {RETRY_DELAY}s... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY)
            return call_groq_api(messages, temperature, max_tokens, retry_count + 1)
        else:
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "latency_sec": latency,
                "tokens": {},
                "retry_count": retry_count
            }
            
    except Exception as e:
        if retry_count < MAX_RETRIES:
            print(f"    💥 Exception, retrying in {RETRY_DELAY}s... (attempt {retry_count + 1})")
            time.sleep(RETRY_DELAY)
            return call_groq_api(messages, temperature, max_tokens, retry_count + 1)
        else:
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": f"Exception after {retry_count} retries: {str(e)}",
                "latency_sec": 0,
                "tokens": {},
                "retry_count": retry_count
            }

def create_structured_output(question_data, persona_data, question_id, api_response):
    """Create structured JSON output"""
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
                "retry_count": api_response["retry_count"]
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
    """Main function - Continue from ID 201-400"""
    print("🚀 GROQ LLAMA EXPERIMENT - CONTINUE ID 201-400")
    print("=" * 60)
    print(f"⚡ Fast mode: {DELAY_SECONDS}s delay with retry mechanism")
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
    print(f"🎯 Using persona ID {persona_data['id']}: {persona_data['prompt'][:80]}...")
    print(f"📋 Processing questions 201-400")
    print()
    
    # Statistics
    success_count = 0
    error_count = 0
    
    # Process questions 201-400
    start_id = 201
    end_id = 400
    
    for question_id in range(start_id, end_id + 1):
        array_index = question_id - 1  # Convert to 0-based index
        output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{question_id:05d}.json")
        
        # Skip if already exists
        if os.path.exists(output_file):
            print(f"[{question_id:3d}] ⏭️  File already exists, skipping...")
            continue
        
        print(f"[{question_id:3d}] 🔄 Processing GSM8K question {question_id}...")
        
        # Check bounds
        if array_index >= len(gsm8k_data):
            print(f"[{question_id:3d}] ❌ Question index out of range! (max: {len(gsm8k_data)})")
            break
        
        question_data = gsm8k_data[array_index]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API with retry
        api_response = call_groq_api(messages)
        
        # Create structured output
        structured_data = create_structured_output(question_data, persona_data, question_id, api_response)
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
            if api_response["success"]:
                retry_info = f" (retries: {api_response['retry_count']})" if api_response['retry_count'] > 0 else ""
                print(f"[{question_id:3d}] ✅ Success (latency: {api_response['latency_sec']:.2f}s{retry_info})")
                success_count += 1
            else:
                print(f"[{question_id:3d}] ❌ Failed: {api_response['error']}")
                error_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 20 questions
        if (question_id - start_id + 1) % 20 == 0:
            total = success_count + error_count
            rate = success_count/total*100 if total > 0 else 0
            progress = (question_id - start_id + 1) / (end_id - start_id + 1) * 100
            print(f"[{question_id:3d}] 📊 Progress: {progress:.1f}% | Success: {success_count} | Errors: {error_count} | Rate: {rate:.1f}%")
        
        # Fast delay
        if question_id < end_id:
            print(f"[{question_id:3d}] ⏳ Waiting {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)
    
    # Final summary
    total = success_count + error_count
    rate = success_count/total*100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 BATCH 201-400 COMPLETED!")
    print(f"📊 Total processed: {total}/200")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success rate: {rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()