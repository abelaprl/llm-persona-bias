#!/usr/bin/env python3
"""
Test 5 new models with 1 man_explicits question
Check if model has reasoning capability
Save to: results/{model_id}/man_explicit/
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime

# New models to test
MODELS_TO_TEST = [
    {
        "order": 31,
        "name": "OpenAI: GPT-OSS-20B",
        "model_id": "openai/gpt-oss-20b",
        "api_key": "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",
        "provider": "openrouter"
    },
    {
        "order": 32,
        "name": "Qwen: Qwen3 Next 80B A3B (Thinking)",
        "model_id": "qwen/qwen3-next-80b-a3b-thinking",
        "api_key": "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",
        "provider": "openrouter"
    },
    {
        "order": 33,
        "name": "Qwen: Qwen3 Next 80B A3B (Instruct)",
        "model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "api_key": "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",
        "provider": "openrouter"
    },
    {
        "order": 34,
        "name": "OpenAI: GPT-OSS-120B",
        "model_id": "openai/gpt-oss-120b",
        "api_key": "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",
        "provider": "openrouter"
    },
    {
        "order": 35,
        "name": "Meta: Llama 3.3 70B Instruct",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "api_key": "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",
        "provider": "openrouter"
    }
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
BASE_RESULTS_DIR = Path("results")

def load_prompts():
    """Load man explicit prompts"""
    prompts_path = Path("inputs/explicits_man_prompts.json")
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

def load_gsm8k_questions(limit=1):
    """Load first N GSM8K questions"""
    questions_path = Path("data/gsm8k_test.json")
    with open(questions_path, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    return all_questions[:limit]

def detect_reasoning(response_text):
    """Detect if response contains reasoning/thinking tags"""
    has_reasoning = False
    reasoning_tags = ['<think>', '<reasoning>', '<analysis>', '###REASONING###']
    
    for tag in reasoning_tags:
        if tag.lower() in response_text.lower():
            has_reasoning = True
            break
    
    return has_reasoning

def call_openrouter_api(model_id, api_key, messages, attempt=1):
    """Call OpenRouter API with retry logic"""
    MAX_RETRIES = 3
    
    for retry in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/abelaprl/TugasAkhir"
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
                content = response.json()['choices'][0]['message']['content']
                has_reasoning = detect_reasoning(content)
                
                return {
                    "success": True,
                    "content": content,
                    "has_reasoning": has_reasoning,
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
                        "has_reasoning": False,
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
                    "has_reasoning": False,
                    "attempts": MAX_RETRIES
                }
    
    return {
        "success": False,
        "error": "All retries exhausted",
        "has_reasoning": False,
        "attempts": MAX_RETRIES
    }

def create_structured_output(question_data, persona_data, question_idx, api_response, model_name, model_id):
    """Create structured JSON output with reasoning detection"""
    safe_model_id = model_id.replace('/', '_').replace(':', '_')
    
    return {
        "metadata": {
            "run_id": f"test_{safe_model_id}_{question_idx:03d}",
            "experiment_type": "new_models_test",
            "model_info": {
                "name": model_name,
                "model_id": model_id,
                "has_reasoning_capability": api_response.get("has_reasoning", False)
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
            "has_reasoning": api_response.get("has_reasoning", False),
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
    print("🧪 TESTING NEW MODELS (1 man_explicit question each)")
    print("="*80 + "\n")
    
    # Load data
    man_explicit_prompts = load_prompts()
    gsm8k_questions = load_gsm8k_questions(limit=1)
    
    print(f"✅ Loaded {len(man_explicit_prompts)} man explicit personas")
    print(f"✅ Loaded {len(gsm8k_questions)} GSM8K question(s)\n")
    
    # Use first persona
    persona = man_explicit_prompts[0]
    question = gsm8k_questions[0]
    
    results_summary = []
    
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
        
        print("[1/1] Processing question...")
        
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
            1,
            api_response,
            model_name,
            model_id
        )
        
        # Save to file
        output_file = model_dir / f"gsm8k_00001.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Status
        if api_response.get("success"):
            latency = api_response.get("latency_sec", 0)
            attempts = api_response.get("attempts", 1)
            has_reasoning = api_response.get("has_reasoning", False)
            
            reasoning_str = "✓ HAS REASONING" if has_reasoning else "✗ NO REASONING"
            print(f"    ✅ Success")
            print(f"    ⏱️  Latency: {latency:.2f}s")
            print(f"    🔄 Attempts: {attempts}")
            print(f"    🧠 {reasoning_str}")
            
            results_summary.append({
                "model": model_name,
                "status": "✅",
                "latency": f"{latency:.2f}s",
                "reasoning": has_reasoning
            })
        else:
            error = api_response.get("error", "Unknown error")
            print(f"    ❌ Failed: {error}")
            
            results_summary.append({
                "model": model_name,
                "status": "❌",
                "latency": "N/A",
                "reasoning": False
            })
        
        print(f"    📁 Results saved to: {model_dir}")
        
        time.sleep(2)  # Rate limiting between models
    
    # Final Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80 + "\n")
    
    for result in results_summary:
        reasoning_indicator = "🧠" if result["reasoning"] else "  "
        print(f"{result['status']} {reasoning_indicator} {result['model']:<45} | Latency: {result['latency']}")
    
    successful = sum(1 for r in results_summary if "✅" in r['status'])
    print(f"\n✅ {successful}/{len(results_summary)} models working")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
