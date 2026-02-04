#!/usr/bin/env python3
"""
Test Google Gemma models: 3-12B, 3-4B, 3n-2B
Run man_explicits 5 questions on each model
Create folder structure: results/{model_id}/{persona}/
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime

# Models to test
MODELS_TO_TEST = [
    {
        "order": 23,
        "name": "Google Gemma 3 12B",
        "model_id": "google/gemma-3-12b-it:free",
        "api_key": "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"
    },
    {
        "order": 27,
        "name": "Google Gemma 3 4B",
        "model_id": "google/gemma-3-4b-it:free",
        "api_key": "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"
    },
    {
        "order": 28,
        "name": "Google Gemma 3n 2B",
        "model_id": "google/gemma-3n-e2b-it:free",
        "api_key": "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"
    }
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
BASE_RESULTS_DIR = Path("results")
DELAY_SECONDS = 2

def load_prompts():
    """Load man explicit prompts"""
    prompts_path = Path("inputs/explicits_man_prompts.json")
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

def load_gsm8k_questions(limit=5):
    """Load first N GSM8K questions"""
    questions_path = Path("data/gsm8k_test.json")
    with open(questions_path, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    return all_questions[:limit]

def call_openrouter_api(model_id, api_key, messages, attempt=1):
    """Call OpenRouter API with retry logic"""
    MAX_RETRIES = 3
    
    for retry in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_id,
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
                error_msg = response.json().get('error', {}).get('message', response.text)
                if retry < MAX_RETRIES:
                    wait_time = 2 ** retry
                    print(f"    ⚠️  Attempt {retry} failed: {error_msg[:80]}")
                    print(f"    ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"{response.status_code}: {error_msg[:100]}",
                        "attempts": retry
                    }
        except Exception as e:
            if retry < MAX_RETRIES:
                wait_time = 2 ** retry
                print(f"    ⚠️  Attempt {retry} error: {str(e)[:80]}")
                print(f"    ⏳ Waiting {wait_time}s before retry...")
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

def create_structured_output(question_data, persona_data, question_idx, api_response, model_name, model_id):
    """Create structured JSON output"""
    safe_model_id = model_id.replace('/', '_').replace(':', '_')
    return {
        "metadata": {
            "run_id": f"test_{safe_model_id}_{question_idx:03d}",
            "experiment_type": "gemma_test",
            "model_info": {
                "name": model_name,
                "model_id": model_id
            },
            "timestamp": datetime.now().isoformat()
        },
        "input": {
            "question_id": question_idx,
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
                "model": model_id
            }
        },
        "response": {
            "success": api_response.get("success"),
            "content": api_response.get("content"),
            "error": api_response.get("error"),
            "raw_response": api_response.get("raw_response"),
            "performance": {
                "latency_sec": api_response.get("latency_sec"),
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

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("🧪 TESTING GOOGLE GEMMA MODELS")
    print("="*80 + "\n")
    
    # Load data
    man_explicit_prompts = load_prompts()
    gsm8k_questions = load_gsm8k_questions(limit=5)
    
    print(f"✅ Loaded {len(man_explicit_prompts)} man explicit personas")
    print(f"✅ Loaded {len(gsm8k_questions)} GSM8K questions\n")
    
    # Test each model
    for model_info in MODELS_TO_TEST:
        model_id = model_info["model_id"]
        model_name = model_info["name"]
        api_key = model_info["api_key"]
        
        print("\n" + "="*80)
        print(f"🚀 TESTING: {model_name}")
        print(f"   Model ID: {model_id}")
        print("="*80 + "\n")
        
        # Create output directory
        safe_model_id = model_id.replace('/', '_').replace(':', '_')
        model_dir = BASE_RESULTS_DIR / safe_model_id / "man_explicit"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Use first persona
        persona = man_explicit_prompts[0]
        
        # Process 5 questions
        results = []
        for q_idx, question in enumerate(gsm8k_questions, 1):
            print(f"[{q_idx}/5] Processing question {q_idx}...")
            
            messages = [
                {"role": "system", "content": persona["prompt"]},
                {"role": "user", "content": question["question"]}
            ]
            
            # Call API
            api_response = call_openrouter_api(model_id, api_key, messages)
            
            # Create output
            output = create_structured_output(
                question,
                persona,
                q_idx,
                api_response,
                model_name,
                model_id
            )
            
            # Save to file
            output_file = model_dir / f"gsm8k_{q_idx:05d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            results.append(output)
            
            # Status
            if api_response.get("success"):
                latency = api_response.get("latency_sec", 0)
                attempts = api_response.get("attempts", 1)
                print(f"    ✅ Success (latency: {latency:.2f}s, attempt: {attempts})")
            else:
                error = api_response.get("error", "Unknown error")
                print(f"    ❌ Failed: {error}")
            
            time.sleep(DELAY_SECONDS)
        
        # Summary for this model
        successful = sum(1 for r in results if r["response"]["success"])
        failed = len(results) - successful
        
        print(f"\n📊 {model_name} Summary:")
        print(f"   ✅ Successful: {successful}/5")
        print(f"   ❌ Failed: {failed}/5")
        print(f"   📁 Results saved to: {model_dir}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETED")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
