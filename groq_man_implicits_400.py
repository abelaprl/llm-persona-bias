#!/usr/bin/env python3
"""
Groq experiment for MAN IMPLICITS (1-400) with warmup
Using woman implicit prompt ID 1 for warmup questions
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
DELAY_SECONDS = 5  # Fast delay
MAX_RETRIES = 3

# Paths
MAN_IMPLICIT_PROMPTS_PATH = "inputs/implicits_man_prompts.json"
WOMAN_IMPLICIT_PROMPTS_PATH = "inputs/implicits_woman_promt.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/man_implicits"

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
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < MAX_RETRIES - 1:
                    print(f"    🔄 Attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                return {
                    "success": False,
                    "content": None,
                    "raw_api_response": None,
                    "error": error_msg,
                    "latency_sec": latency,
                    "tokens": {},
                    "attempt": attempt + 1
                }
                
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            if attempt < MAX_RETRIES - 1:
                print(f"    🔄 Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": error_msg,
                "latency_sec": 0,
                "tokens": {},
                "attempt": attempt + 1
            }

def create_structured_output(question_data, persona_data, question_id, api_response, persona_type="man_implicit"):
    """Create structured JSON output"""
    return {
        "metadata": {
            "run_id": f"groq_{persona_type}_{persona_data['id']:03d}_{question_id:05d}",
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
                "type": persona_type,
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
                "attempts": api_response["attempt"]
            }
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
            "reasoning_quality": None,
            "bias_indicators": None
        }
    }

def run_warmup(woman_persona_data, gsm8k_data):
    """Run 3 warmup questions with woman implicit persona"""
    print("🔥 WARMUP: Running 3 questions with woman implicit persona...")
    
    warmup_dir = os.path.join(OUTPUT_DIR, "warmup")
    os.makedirs(warmup_dir, exist_ok=True)
    
    for i in range(3):
        question_id = i + 1
        print(f"  🔥 Warmup {question_id}/3...")
        
        question_data = gsm8k_data[i]
        
        messages = [
            {"role": "system", "content": woman_persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        api_response = call_groq_api(messages)
        
        # Save warmup result
        warmup_output = create_structured_output(
            question_data, woman_persona_data, question_id, api_response, "woman_implicit_warmup"
        )
        
        warmup_file = os.path.join(warmup_dir, f"warmup_{question_id:03d}.json")
        with open(warmup_file, "w", encoding="utf-8") as f:
            json.dump(warmup_output, f, indent=2, ensure_ascii=False)
        
        if api_response["success"]:
            print(f"  🔥 Warmup {question_id}/3: ✅ Success ({api_response['latency_sec']:.2f}s)")
        else:
            print(f"  🔥 Warmup {question_id}/3: ❌ {api_response['error']}")
        
        time.sleep(2)  # Short delay between warmup
    
    print("🔥 Warmup completed! Starting main experiment...\n")

def main():
    """Main function for man implicits with warmup"""
    print("🚀 GROQ LLAMA EXPERIMENT - MAN IMPLICITS (1-400) WITH WARMUP")
    print("=" * 70)
    print(f"🔥 Warmup: 3 questions with woman implicit persona ID 1")
    print(f"🎯 Main: 400 questions with man implicit persona ID 1")
    print(f"⏳ Delay: {DELAY_SECONDS}s between requests")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    with open(MAN_IMPLICIT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        man_prompts = json.load(f)
    
    with open(WOMAN_IMPLICIT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        woman_prompts = json.load(f)
    
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    print(f"✅ Loaded {len(man_prompts)} man implicit prompts")
    print(f"✅ Loaded {len(woman_prompts)} woman implicit prompts")
    print(f"✅ Loaded {len(gsm8k_data)} GSM8K questions")
    
    # Use first prompts (ID 1)
    man_persona = man_prompts[0]  # ID 1
    woman_persona = woman_prompts[0]  # ID 1
    
    print(f"🎯 Man persona ID {man_persona['id']}: {man_persona['prompt'][:80]}...")
    print(f"🔥 Woman persona ID {woman_persona['id']}: {woman_persona['prompt'][:80]}...")
    print()
    
    # Run warmup
    run_warmup(woman_persona, gsm8k_data)
    
    # Statistics
    success_count = 0
    error_count = 0
    
    # Process questions 1-400
    for i in range(400):
        question_id = i + 1
        output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{question_id:05d}.json")
        
        # Skip if already exists
        if os.path.exists(output_file):
            print(f"[{question_id:3d}] ⏭️  File already exists, skipping...")
            continue
        
        print(f"[{question_id:3d}] 🔄 Processing GSM8K question {question_id}...")
        
        # Check bounds
        if i >= len(gsm8k_data):
            print(f"[{question_id:3d}] ❌ Question index out of range!")
            break
        
        question_data = gsm8k_data[i]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": man_persona["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API with retry
        api_response = call_groq_api(messages)
        
        # Create structured output
        structured_data = create_structured_output(question_data, man_persona, question_id, api_response)
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
            if api_response["success"]:
                print(f"[{question_id:3d}] ✅ Success (latency: {api_response['latency_sec']:.2f}s, attempt: {api_response['attempt']})")
                success_count += 1
            else:
                print(f"[{question_id:3d}] ❌ Failed after {api_response['attempt']} attempts: {api_response['error']}")
                error_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 20 questions
        if question_id % 20 == 0:
            total = success_count + error_count
            rate = success_count/total*100 if total > 0 else 0
            print(f"[{question_id:3d}] 📊 Progress: {question_id}/400 | Success: {success_count} | Errors: {error_count} | Rate: {rate:.1f}%")
        
        # Rate limiting
        if question_id < 400:
            time.sleep(DELAY_SECONDS)
    
    # Final summary
    total = success_count + error_count
    rate = success_count/total*100 if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("🎉 MAN IMPLICITS EXPERIMENT COMPLETED!")
    print(f"📊 Total processed: {total}/400")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success rate: {rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"🔥 Warmup files in: {OUTPUT_DIR}/warmup/")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()