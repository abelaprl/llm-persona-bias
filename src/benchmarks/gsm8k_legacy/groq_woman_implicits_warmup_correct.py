#!/usr/bin/env python3
"""
Run woman_implicits experiment 1-400 GSM8K questions
Using Groq Llama 3.1 8B Instant
With warmup using implicit persona
"""

import json
import requests
import time
import random
from pathlib import Path
from datetime import datetime

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"
DELAY_SECONDS = 3
MAX_RETRIES = 5

def load_data():
    """Load all required data"""
    # Load woman implicit prompts
    with open("inputs/implicits_woman_promt.json", 'r', encoding='utf-8') as f:
        woman_implicit_prompts = json.load(f)
    
    # Load GSM8K questions
    with open("data/gsm8k/gsm8k_test.json", 'r', encoding='utf-8') as f:
        gsm8k_questions = json.load(f)
    
    return woman_implicit_prompts, gsm8k_questions

def call_groq_api(messages, attempt=1):
    """Call Groq API with retry logic"""
    for retry in range(1, MAX_RETRIES + 1):
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
                    "temperature": 0.2,
                    "max_tokens": 512
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "content": response.json()['choices'][0]['message']['content'],
                    "raw_response": response.json(),
                    "latency_sec": response.elapsed.total_seconds(),
                    "attempts": retry
                }
            else:
                if retry < MAX_RETRIES:
                    wait_time = 2 ** retry
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"{response.status_code}: {response.text[:100]}",
                        "attempts": retry
                    }
        except Exception as e:
            if retry < MAX_RETRIES:
                wait_time = 2 ** retry
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "error": str(e)[:100],
                    "attempts": retry
                }
    
    return {
        "success": False,
        "error": "All retries exhausted",
        "attempts": MAX_RETRIES
    }

def create_structured_output(question_data, persona_data, question_id, api_response, persona_type="woman_implicit"):
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
            "success": api_response.get("success"),
            "content": api_response.get("content"),
            "raw_api_response": api_response.get("raw_response"),
            "error": api_response.get("error"),
            "performance": {
                "latency_sec": api_response.get("latency_sec"),
                "tokens": api_response.get("raw_response", {}).get("usage", {}) if api_response.get("raw_response") else None,
                "attempts": api_response.get("attempts")
            }
        },
        "analysis": {
            "extracted_answer": None,
            "is_correct": None,
            "reasoning_quality": None,
            "bias_indicators": None
        }
    }

def run_warmup(persona_data, warmup_questions):
    """Run warmup with 1 implicit persona question"""
    print("\n🔥 WARMUP PHASE")
    print("="*60)
    
    # Use first warmup question
    warmup_q = warmup_questions[0]
    
    messages = [
        {"role": "system", "content": persona_data["prompt"]},
        {"role": "user", "content": warmup_q}
    ]
    
    print(f"[Warmup] Running 1 warmup question from woman implicit persona ID 1...")
    api_response = call_groq_api(messages)
    
    if api_response.get("success"):
        latency = api_response.get("latency_sec", 0)
        print(f"✅ Warmup complete (latency: {latency:.2f}s)")
        
        # Save warmup to file
        warmup_output_dir = Path("results/groq-llama-3.1-8b-instant/woman_implicits/warmup")
        warmup_output_dir.mkdir(parents=True, exist_ok=True)
        
        warmup_data = {
            "metadata": {
                "type": "warmup",
                "persona_id": persona_data["id"],
                "persona_type": "woman_implicit",
                "timestamp": datetime.now().isoformat()
            },
            "warmup_question": warmup_q,
            "warmup_response": api_response.get("content"),
            "latency_sec": latency
        }
        
        warmup_file = warmup_output_dir / "warmup_001.json"
        with open(warmup_file, 'w', encoding='utf-8') as f:
            json.dump(warmup_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Warmup saved to: {warmup_file}")
    else:
        print(f"❌ Warmup failed: {api_response.get('error')}")
    
    time.sleep(DELAY_SECONDS)

def main():
    """Main experiment function"""
    print("\n" + "="*80)
    print("🚀 WOMAN IMPLICITS EXPERIMENT (400 Questions)")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Temperature: 0.2")
    print(f"Delay: {DELAY_SECONDS}s between requests")
    print("="*80 + "\n")
    
    # Load data
    woman_implicit_prompts, gsm8k_questions = load_data()
    
    print(f"✅ Loaded {len(woman_implicit_prompts)} woman implicit personas")
    print(f"✅ Loaded {len(gsm8k_questions)} GSM8K questions")
    
    # Create output directory
    output_dir = Path("results/groq-llama-3.1-8b-instant/woman_implicits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use first persona (ID 1)
    persona = woman_implicit_prompts[0]
    print(f"\n📍 Using persona ID: {persona['id']}")
    print(f"📝 Persona: {persona['prompt'][:100]}...\n")
    
    # Load warmup questions (reuse from implicits_woman)
    with open("inputs/implicits_woman_promt.json", 'r', encoding='utf-8') as f:
        warmup_prompts = [p["prompt"] for p in json.load(f)]
    
    # Run warmup
    run_warmup(persona, warmup_prompts)
    
    # Main experiment
    print("\n🧮 MAIN EXPERIMENT (GSM8K Questions)")
    print("="*60)
    
    successful = 0
    errors = 0
    start_time = datetime.now()
    
    for q_idx in range(1, min(401, len(gsm8k_questions) + 1)):
        question = gsm8k_questions[q_idx - 1]
        
        # Call API
        messages = [
            {"role": "system", "content": persona["prompt"]},
            {"role": "user", "content": question["question"]}
        ]
        
        api_response = call_groq_api(messages)
        
        # Create and save output
        output_data = create_structured_output(
            question,
            persona,
            q_idx,
            api_response
        )
        
        output_file = output_dir / f"gsm8k_{q_idx:05d}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Status update
        if api_response.get("success"):
            successful += 1
            latency = api_response.get("latency_sec", 0)
            attempts = api_response.get("attempts", 1)
            print(f"[{q_idx:3d}] ✅ Success (latency: {latency:.2f}s, attempt: {attempts})")
        else:
            errors += 1
            error = api_response.get("error", "Unknown error")
            print(f"[{q_idx:3d}] ❌ Failed: {error}")
        
        # Progress update every 20 questions
        if q_idx % 20 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = successful / (successful + errors) * 100 if (successful + errors) > 0 else 0
            print(f"\n📊 Progress: {q_idx}/400 | Success: {successful} | Errors: {errors} | Rate: {rate:.1f}%\n")
        
        time.sleep(DELAY_SECONDS)
    
    # Final summary
    total = successful + errors
    success_rate = successful / total * 100 if total > 0 else 0
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("🎉 WOMAN IMPLICITS EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"📊 Total processed: {total}/400")
    print(f"✅ Successful: {successful}")
    print(f"❌ Errors: {errors}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"📁 Results saved to: {output_dir}")
    print(f"⏰ Duration: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
