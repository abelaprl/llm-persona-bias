#!/usr/bin/env python3
"""
Run woman implicits experiment: 400 GSM8K questions with woman implicit persona
Uses Groq API directly with temperature 0.2 for reproducibility
Includes 1 warmup question at the start, then 400 main GSM8K questions
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
DELAY_SECONDS = 5
MAX_RETRIES = 5

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

def create_structured_output(question_data, persona_data, question_id, api_response, is_warmup=False):
    """Create structured JSON output"""
    if is_warmup:
        content = api_response['choices'][0]['message']['content']
        return {
            "metadata": {
                "run_id": f"groq_woman_implicit_001_warmup_001",
                "experiment_type": "gsm8k_bias_study",
                "question_source": "warmup",
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "provider": "groq",
                    "model_id": MODEL_ID,
                    "api_endpoint": GROQ_API_URL
                }
            },
            "input": {
                "question_id": 1,
                "dataset": "warmup",
                "question": persona_data["prompt"],
                "ground_truth": "Warmup question response",
                "persona": {
                    "id": persona_data["id"],
                    "type": "woman_implicit",
                    "prompt": persona_data["prompt"]
                }
            },
            "request": {
                "messages": [
                    {"role": "system", "content": persona_data["prompt"]},
                    {"role": "user", "content": persona_data["prompt"]}
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
    else:
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

def run_warmup(persona_data):
    """Run single warmup question"""
    print("\n" + "="*70)
    print("🔥 WARMUP PHASE")
    print("="*70)
    print(f"Persona ID: {persona_data['id']} (woman implicit)")
    
    messages = [
        {"role": "system", "content": persona_data["prompt"]},
        {"role": "user", "content": persona_data["prompt"]}
    ]
    
    print(f"⏳ Sending warmup question...", end=" ")
    start_time = time.time()
    api_response, attempt = call_groq_api(messages)
    elapsed = time.time() - start_time
    
    if api_response:
        output_data = create_structured_output(None, persona_data, 1, api_response, is_warmup=True)
        warmup_file = WARMUP_PATH / "warmup_001.json"
        with open(warmup_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Success (latency: {elapsed:.2f}s, attempt: {attempt})")
        return True
    else:
        print(f"❌ Failed after {MAX_RETRIES} attempts")
        return False

def main():
    """Main experiment loop"""
    print("\n" + "="*70)
    print("🚀 WOMAN IMPLICITS EXPERIMENT (400 QUESTIONS)")
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
    
    # Run warmup
    warmup_ok = run_warmup(persona_data)
    if not warmup_ok:
        print("⚠️  Warmup failed, but continuing with main experiment...")
    
    # Main experiment
    print("\n" + "="*70)
    print("🧮 MAIN EXPERIMENT PHASE (400 Questions)")
    print("="*70)
    
    successful = 0
    errors = 0
    start_time = time.time()
    
    for idx, question_data in enumerate(gsm8k_data[:400], 1):
        question_id = idx  # Sequential ID from enumeration
        
        # Show progress at every 20 questions
        if idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = 100.0 * successful / idx
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
            output_data = create_structured_output(question_data, persona_data, question_id, api_response, is_warmup=False)
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
    success_rate = 100.0 * successful / 400
    
    print("\n" + "="*70)
    print("🎉 WOMAN IMPLICITS EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"📊 Total processed: {successful + errors}/400")
    print(f"✅ Successful: {successful}")
    print(f"❌ Errors: {errors}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"⏱️  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}m)")
    print(f"📁 Results saved to: {RESULTS_PATH}")
    print(f"🔥 Warmup file: {WARMUP_PATH / 'warmup_001.json'}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
