#!/usr/bin/env python3
"""
Continue woman implicits experiment: 400 GSM8K questions with woman implicit persona
Resumes from question 279 (if you want to start fresh, change START_QUESTION)
"""

import json
import time
import random
import requests
from datetime import datetime
from pathlib import Path

# Configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"
TEMPERATURE = 0.2
MAX_TOKENS = 512
DELAY_SECONDS = 5  # Increased delay to avoid rate limiting
MAX_RETRIES = 5
START_QUESTION = 279  # Continue from here

# Paths
BASE_PATH = Path(__file__).parent
INPUTS_PATH = BASE_PATH / "inputs"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results" / "groq-llama-3.1-8b-instant" / "woman_implicits"
WARMUP_PATH = RESULTS_PATH / "warmup"

# Create directories
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
WARMUP_PATH.mkdir(parents=True, exist_ok=True)

def call_groq_api(messages):
    """
    Call Groq API with retry mechanism and exponential backoff
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_ID,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), attempt
            elif response.status_code == 429:
                # Rate limit - exponential backoff
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"    ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠️ API error (attempt {attempt}/{MAX_RETRIES}): {response.status_code}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    
        except requests.exceptions.Timeout:
            print(f"    ⏱️ Timeout (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    ❌ Error (attempt {attempt}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    
    return None, MAX_RETRIES

def load_personas():
    """Load woman implicit personas from inputs"""
    persona_file = INPUTS_PATH / "implicits_woman_promt.json"
    with open(persona_file, 'r', encoding='utf-8') as f:
        personas = json.load(f)
    return personas

def load_gsm8k_data():
    """Load GSM8K test data"""
    data_file = DATA_PATH / "gsm8k_test.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_structured_output(question_data, persona_data, question_id, api_response):
    """Create structured JSON output"""
    content = api_response['choices'][0]['message']['content']
    return {
        "metadata": {
            "run_id": f"groq_woman_implicit_001_{question_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "question_source": "gsm8k",
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
                "type": "woman_implicit",
                "prompt": persona_data["prompt"]
            }
        },
        "request": {
            "messages": [
                {"role": "system", "content": persona_data["prompt"]},
                {"role": "user", "content": question_data["question"]}
            ],
            "parameters": {
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "model": MODEL_ID
            }
        },
        "response": {
            "success": True,
            "content": content,
            "raw_api_response": api_response,
            "error": None,
            "performance": {
                "latency_sec": api_response['usage']['total_time'],
                "tokens": api_response['usage'],
                "attempts": 1
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
    """Main experiment loop"""
    print("\n" + "="*70)
    print(f"🚀 WOMAN IMPLICITS EXPERIMENT (CONTINUE FROM Q{START_QUESTION})")
    print("="*70)
    print(f"📊 Model: {MODEL_ID}")
    print(f"🌡️  Temperature: {TEMPERATURE}")
    print(f"⏱️  Delay between requests: {DELAY_SECONDS}s")
    print(f"🔄 Max retries: {MAX_RETRIES}")
    
    # Load data
    print("\n📂 Loading data...")
    personas = load_personas()
    gsm8k_data = load_gsm8k_data()
    
    # Use persona ID 1
    persona_id = 1
    persona_data = personas[persona_id - 1]  # 0-indexed
    print(f"✅ Loaded persona ID {persona_id} (woman implicit)")
    print(f"✅ Loaded {len(gsm8k_data)} GSM8K questions")
    
    # Main experiment
    print("\n" + "="*70)
    print(f"🧮 CONTINUING FROM QUESTION {START_QUESTION}/400")
    print("="*70)
    
    successful = 0
    errors = 0
    start_time = time.time()
    
    for idx, question_data in enumerate(gsm8k_data[START_QUESTION-1:400], START_QUESTION):
        question_id = idx
        
        # Show progress at every 20 questions
        if idx % 20 == 0:
            elapsed = time.time() - start_time
            total_processed = idx - START_QUESTION + 1
            if total_processed > 0:
                rate = 100.0 * successful / total_processed
                print(f"📊 Progress: {idx}/400 | Success: {successful} | Errors: {errors} | Rate: {rate:.1f}%")
        
        # Create messages with persona context
        messages = [
            {"role": "system", "content": persona_data["prompt"]},
            {"role": "user", "content": question_data["question"]}
        ]
        
        # API call
        api_response, attempt = call_groq_api(messages)
        
        if api_response:
            # Success
            output_data = create_structured_output(question_data, persona_data, question_id, api_response)
            output_file = RESULTS_PATH / f"gsm8k_{question_id:05d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            latency = api_response['usage']['total_time']
            print(f"[{idx:3d}] ✅ Success (latency: {latency:.2f}s, attempt: {attempt})")
            successful += 1
        else:
            # Failed
            errors += 1
            print(f"[{idx:3d}] ❌ Failed after {MAX_RETRIES} attempts")
        
        # Rate limiting delay
        if idx < 400:
            time.sleep(DELAY_SECONDS)
    
    # Final summary
    elapsed_total = time.time() - start_time
    total_processed = 400 - START_QUESTION + 1
    success_rate = 100.0 * successful / total_processed if total_processed > 0 else 0
    
    print("\n" + "="*70)
    print("🎉 WOMAN IMPLICITS EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"📊 Questions {START_QUESTION}-400 processed: {successful + errors}/{total_processed}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Errors: {errors}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"⏱️  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")
    print(f"📁 Results saved to: {RESULTS_PATH}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
