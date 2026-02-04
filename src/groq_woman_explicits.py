#!/usr/bin/env python3
"""
Groq experiment untuk woman explicits dengan ID 1-200
Menggunakan struktur JSON yang sama dengan man explicits
"""

import json
import os
import time
import requests
from datetime import datetime
import re

# Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"
BASE_DELAY = 3  # Base delay in seconds
MAX_RETRIES = 3

# Paths
INPUT_PROMPTS_PATH = "inputs/explicits_woman_promts.json"
GSM8K_DATA_PATH = "data/gsm8k_test.json"
OUTPUT_DIR = "results/groq-llama-3.1-8b-instant/woman_explicits"

def call_groq_api(messages, temperature=0.2, max_tokens=512, retry=0):
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
                "retry_count": retry
            }
        elif response.status_code == 429 and retry < MAX_RETRIES:
            # Rate limit hit, retry with exponential backoff
            wait_time = (2 ** retry) * 10  # 10s, 20s, 40s
            print(f"    🔄 Rate limit hit, retrying in {wait_time}s (attempt {retry + 1}/{MAX_RETRIES + 1})...")
            time.sleep(wait_time)
            return call_groq_api(messages, temperature, max_tokens, retry + 1)
        else:
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "latency_sec": latency,
                "tokens": {},
                "retry_count": retry
            }
            
    except Exception as e:
        if retry < MAX_RETRIES:
            wait_time = (2 ** retry) * 5  # 5s, 10s, 20s
            print(f"    🔄 Exception occurred, retrying in {wait_time}s (attempt {retry + 1}/{MAX_RETRIES + 1})...")
            time.sleep(wait_time)
            return call_groq_api(messages, temperature, max_tokens, retry + 1)
        else:
            return {
                "success": False,
                "content": None,
                "raw_api_response": None,
                "error": f"Exception after {retry + 1} attempts: {str(e)}",
                "latency_sec": 0,
                "tokens": {},
                "retry_count": retry
            }

def extract_numerical_answer(text):
    """Extract numerical answer from response text"""
    if not text:
        return None
    
    # Look for patterns like "$18", "18 dollars", "answer is 18", etc.
    patterns = [
        r'\$(\d+(?:\.\d+)?)',                    # $18
        r'(\d+(?:\.\d+)?) dollars?',             # 18 dollars
        r'answer is (\d+(?:\.\d+)?)',            # answer is 18
        r'makes (\d+(?:\.\d+)?)',                # makes 18
        r'total of \$?(\d+(?:\.\d+)?)',          # total of $18
        r'= \$?(\d+(?:\.\d+)?)',                 # = $18
        r'(\d+(?:\.\d+)?)$'                      # 18 at end
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return float(matches[-1])  # Return last match
            except:
                continue
    
    return None

def extract_ground_truth_answer(ground_truth):
    """Extract numerical answer from ground truth"""
    if not ground_truth:
        return None
    
    # Look for #### pattern
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', ground_truth)
    if match:
        return float(match.group(1))
    
    # Fallback to other patterns
    patterns = [r'\$(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?) dollar']
    for pattern in patterns:
        matches = re.findall(pattern, ground_truth)
        if matches:
            try:
                return float(matches[-1])
            except:
                continue
    
    return None

def create_structured_output(question_data, persona_data, question_id, api_response):
    """Create structured JSON output with analysis"""
    # Extract answers
    extracted_answer = extract_numerical_answer(api_response["content"]) if api_response["content"] else None
    ground_truth_answer = extract_ground_truth_answer(question_data["answer"])
    
    # Check correctness
    is_correct = None
    if extracted_answer is not None and ground_truth_answer is not None:
        is_correct = abs(extracted_answer - ground_truth_answer) < 0.01
    
    # Assess reasoning quality
    reasoning_quality = None
    if api_response["content"]:
        content_length = len(api_response["content"])
        if content_length > 200:
            reasoning_quality = "detailed"
        elif content_length > 50:
            reasoning_quality = "brief"
        else:
            reasoning_quality = "minimal"
    
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
                "retry_count": api_response["retry_count"]
            }
        },
        "analysis": {
            "extracted_answer": extracted_answer,
            "ground_truth_answer": ground_truth_answer,
            "is_correct": is_correct,
            "reasoning_quality": reasoning_quality,
            "bias_indicators": None,
            "analysis_timestamp": datetime.now().isoformat()
        }
    }

def main():
    """Main function for woman explicits experiment"""
    print("🚀 GROQ LLAMA EXPERIMENT - WOMAN EXPLICITS (ID 1-200)")
    print("=" * 60)
    print(f"👩 Target: Woman explicit personas")
    print(f"📊 Range: Questions 1-200")
    print(f"⏳ Delay: {BASE_DELAY}s base + exponential backoff on retry")
    print(f"🔄 Max retries: {MAX_RETRIES}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    with open(INPUT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    with open(GSM8K_DATA_PATH, "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    
    print(f"✅ Loaded {len(prompts)} woman explicit prompts and {len(gsm8k_data)} GSM8K questions")
    
    # Use first prompt (ID 1)
    persona_data = prompts[0]
    print(f"👩 Using persona ID {persona_data['id']}: {persona_data['prompt'][:80]}...")
    print()
    
    # Statistics
    success_count = 0
    error_count = 0
    correct_count = 0
    
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
            {"role": "system", "content": persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # Call API with retries
        api_response = call_groq_api(messages)
        
        # Create structured output with analysis
        structured_data = create_structured_output(question_data, persona_data, question_id, api_response)
        
        # Save result
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
            if api_response["success"]:
                print(f"[{question_id:3d}] ✅ Success (latency: {api_response['latency_sec']:.2f}s, retries: {api_response['retry_count']})")
                if structured_data["analysis"]["is_correct"]:
                    correct_count += 1
                    print(f"[{question_id:3d}]    ✓ Correct answer: {structured_data['analysis']['extracted_answer']}")
                else:
                    print(f"[{question_id:3d}]    ✗ Wrong answer: {structured_data['analysis']['extracted_answer']} (expected: {structured_data['analysis']['ground_truth_answer']})")
                success_count += 1
            else:
                print(f"[{question_id:3d}] ❌ Failed after {api_response['retry_count']} retries: {api_response['error']}")
                error_count += 1
                
        except Exception as e:
            print(f"[{question_id:3d}] 💥 Save error: {e}")
            error_count += 1
        
        # Show progress every 10 questions
        if question_id % 10 == 0:
            total = success_count + error_count
            success_rate = success_count/total*100 if total > 0 else 0
            accuracy = correct_count/success_count*100 if success_count > 0 else 0
            print(f"[{question_id:3d}] 📊 Progress: {question_id}/200 | Success: {success_count} | Errors: {error_count} | Correct: {correct_count} | Accuracy: {accuracy:.1f}%")
        
        # Rate limiting
        if question_id < 200:
            time.sleep(BASE_DELAY)
    
    # Final summary
    total = success_count + error_count
    success_rate = success_count/total*100 if total > 0 else 0
    accuracy = correct_count/success_count*100 if success_count > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 WOMAN EXPLICITS EXPERIMENT COMPLETED!")
    print(f"📊 Total processed: {total}/200")
    print(f"✅ Successful API calls: {success_count}")
    print(f"❌ Failed API calls: {error_count}")
    print(f"🎯 Correct answers: {correct_count}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"🧮 Accuracy rate: {accuracy:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()