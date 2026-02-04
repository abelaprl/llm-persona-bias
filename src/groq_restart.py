#!/usr/bin/env python3
"""
Script sederhana untuk menjalankan eksperimen Groq dengan delay panjang
Restart dari ID 1 dengan 15 detik delay per request
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
DELAY_SECONDS = 15  # 15 detik delay per request

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_man_prompts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json" 
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/man_explicits"

def call_groq_api(messages):
    """Call Groq API with error handling"""
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
    
    try:
        start_time = time.time()
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            return response.json(), latency, None
        else:
            return None, latency, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except Exception as e:
        return None, 0, f"Exception: {str(e)}"

def main():
    """Main function with restart option"""
    print("🚀 GROQ LLAMA EXPERIMENT - RESTART FROM ID 1")
    print("=" * 60)
    print(f"⏳ Using {DELAY_SECONDS}s delay per request to avoid rate limits")
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
    base_prompt = prompts[0]["prompt"]
    print(f"🎯 Using persona ID 1: {base_prompt[:80]}...")
    print()
    
    # Ask if restart from scratch
    restart = input("❓ Delete existing files and restart from ID 1? (y/N): ")
    if restart.lower() == 'y':
        import glob
        existing_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
        for file_path in existing_files:
            os.remove(file_path)
        print(f"🗑️  Deleted {len(existing_files)} existing files")
    print()
    
    # Statistics
    success_count = 0
    error_count = 0
    
    # Process questions 1-200
    for i in range(200):
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
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API
        result, latency, error = call_groq_api(messages)
        
        # Prepare output data
        output_data = {
            "run_id": f"groq_man_explicit_001_{question_id:05d}",
            "model_id": f"groq/{MODEL_ID}",
            "order": 0,
            "question_id": question_id,
            "persona_id": 1,
            "persona_prompt": base_prompt,
            "question": question_data["question"],
            "ground_truth": question_data["answer"],
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
            "latency_sec": latency,
            "success": error is None,
            "response": result,
            "error": error
        }
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            if error:
                print(f"[{question_id:3d}] ❌ API Error: {error}")
                error_count += 1
            else:
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"[{question_id:3d}] ✅ Success (latency: {latency:.2f}s)")
                print(f"[{question_id:3d}]    Response: {response_text[:60]}...")
                success_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 10 questions
        if question_id % 10 == 0:
            total = success_count + error_count
            rate = success_count/total*100 if total > 0 else 0
            print(f"[{question_id:3d}] 📊 Progress: {question_id}/200 | Success: {success_count} | Errors: {error_count} | Rate: {rate:.1f}%")
        
        # Long delay to avoid rate limits (except for last question)
        if question_id < 200:
            print(f"[{question_id:3d}] ⏳ Waiting {DELAY_SECONDS}s to avoid rate limits...")
            
            # Show countdown every 5 seconds
            for remaining in range(DELAY_SECONDS, 0, -1):
                if remaining % 5 == 0 or remaining <= 3:
                    print(f"    ⏰ {remaining}s remaining...")
                time.sleep(1)
            print()
    
    # Final summary
    total = success_count + error_count
    rate = success_count/total*100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED!")
    print(f"📊 Total processed: {total}/200")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success rate: {rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()