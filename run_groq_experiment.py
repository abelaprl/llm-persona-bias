#!/usr/bin/env python3
"""
Script untuk menjalankan eksperimen GSM8K dengan Groq Llama 3.1 8B
untuk dataset "man explicits" dengan ID 1-200
"""

import json
import os
import time
import requests
from datetime import datetime

# Konfigurasi
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_man_prompts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/man_explicits"

def load_data():
    """Load input prompts and GSM8K data"""
    # Load man explicit prompts
    with open(INPUT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    # Load GSM8K test data
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    return prompts, gsm8k_data

def call_groq_api(messages, temperature=0.2, max_tokens=500, max_retries=3):
    """Call Groq API with retry mechanism"""
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
                # Rate limit hit, wait longer
                wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                print(f"    ⏳ Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                return None, latency, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10, 20 seconds
                print(f"    ⚠️ API error, retrying in {wait_time}s... ({attempt+1}/{max_retries})")
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

def process_single_case(persona_id, persona_prompt, gsm8k_id, gsm8k_item):
    """Process a single test case"""
    print(f"🧮 Processing Persona {persona_id}, GSM8K {gsm8k_id}...")
    
    messages = create_messages(persona_prompt, gsm8k_item["question"])
    
    # Call API
    result, latency, error = call_groq_api(messages)
    
    if error:
        print(f"  ❌ Error: {error}")
        return {
            "persona_id": persona_id,
            "gsm8k_id": gsm8k_id,
            "timestamp": datetime.now().isoformat(),
            "persona_prompt": persona_prompt,
            "question": gsm8k_item["question"],
            "correct_answer": gsm8k_item["answer"],
            "model_response": None,
            "error": error,
            "latency": latency,
            "success": False
        }
    
    # Extract response
    if result and "choices" in result:
        model_response = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        print(f"  ✅ Success! Latency: {latency:.2f}s")
        
        return {
            "persona_id": persona_id,
            "gsm8k_id": gsm8k_id,
            "timestamp": datetime.now().isoformat(),
            "persona_prompt": persona_prompt,
            "question": gsm8k_item["question"],
            "correct_answer": gsm8k_item["answer"],
            "model_response": model_response,
            "usage": usage,
            "latency": latency,
            "error": None,
            "success": True,
            "raw_result": result
        }
    else:
        print(f"  ❌ No valid response received")
        return {
            "persona_id": persona_id,
            "gsm8k_id": gsm8k_id,
            "timestamp": datetime.now().isoformat(),
            "persona_prompt": persona_prompt,
            "question": gsm8k_item["question"],
            "correct_answer": gsm8k_item["answer"],
            "model_response": None,
            "error": "No valid response",
            "latency": latency,
            "success": False
        }

def save_result(result, persona_id, gsm8k_id):
    """Save individual result to file"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save individual result
    filename = f"gsm8k_{gsm8k_id:05d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def run_experiment(start_id=1, end_id=200):
    """Run the main experiment"""
    print(f"🚀 STARTING GROQ LLAMA EXPERIMENT")
    print(f"📊 Model: {MODEL_ID}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📈 Range: GSM8K ID {start_id} to {end_id}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    prompts, gsm8k_data = load_data()
    
    # Use persona ID 1 (simple "Your user is a man.")
    target_persona = None
    for prompt in prompts:
        if prompt["id"] == 1:
            target_persona = prompt
            break
    
    if not target_persona:
        print("❌ Persona ID 1 not found!")
        return
    
    print(f"👤 Using Persona {target_persona['id']}: {target_persona['prompt']}")
    print()
    
    # Process each GSM8K problem
    success_count = 0
    error_count = 0
    
    for gsm8k_id in range(start_id, end_id + 1):
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
        save_result(result, target_persona["id"], gsm8k_id)
        
        # Update counters
        if result["success"]:
            success_count += 1
        else:
            error_count += 1
        
        # Progress update
        if gsm8k_id % 10 == 0:
            progress = (gsm8k_id - start_id + 1) / (end_id - start_id + 1) * 100
            print(f"📊 Progress: {progress:.1f}% ({gsm8k_id}/{end_id}) | Success: {success_count} | Errors: {error_count}")
        
        # Add delay to avoid rate limiting (Groq has 6000 TPM limit)
        time.sleep(2.0)  # Increased delay to 2 seconds
    
    # Final summary
    total_processed = success_count + error_count
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED!")
    print(f"📊 Total Processed: {total_processed}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Test API connection first
    print("🔧 Testing Groq API connection...")
    test_messages = [{"role": "user", "content": "Hello, just respond with 'OK'"}]
    result, latency, error = call_groq_api(test_messages)
    
    if error:
        print(f"❌ API test failed: {error}")
        exit(1)
    else:
        print(f"✅ API test successful! Latency: {latency:.2f}s")
        print()
    
    # Run main experiment
    run_experiment(start_id=1, end_id=200)