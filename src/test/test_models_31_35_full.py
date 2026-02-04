#!/usr/bin/env python3
"""
Test models 31-35 with GSM8K questions and structured JSON output.
Handles reasoning-enabled models (model 32) with correct parameters.
"""

import json
import os
import time
import requests
from datetime import datetime

# Models configuration
MODELS = [
    {"order": 31, "model_id": "openai/gpt-oss-20b", "folder": "openai_gpt-oss-20b", "has_reasoning": False},
    {"order": 32, "model_id": "qwen/qwen3-next-80b-a3b-thinking", "folder": "qwen_qwen3-next-80b-a3b-thinking", "has_reasoning": True},
    {"order": 33, "model_id": "qwen/qwen3-next-80b-a3b-instruct", "folder": "qwen_qwen3-next-80b-a3b-instruct", "has_reasoning": False},
    {"order": 34, "model_id": "openai/gpt-oss-120b", "folder": "openai_gpt-oss-120b", "has_reasoning": False},
    {"order": 35, "model_id": "meta-llama/llama-3.3-70b-instruct", "folder": "meta-llama_llama-3.3-70b-instruct", "has_reasoning": False},
]

# API configuration
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Paths
PERSONA_PATH = "inputs/explicits_man_prompts.json"
GSM8K_PATH = "data/gsm8k_test.json"

def load_data():
    """Load persona and GSM8K data"""
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        personas = json.load(f)
    
    with open(GSM8K_PATH, "r", encoding="utf-8") as f:
        gsm8k = json.load(f)
    
    return personas, gsm8k

def call_openrouter_api(model_id, messages, has_reasoning=False, max_retries=3):
    """Call OpenRouter API with retry logic"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/user/project",
        "X-Title": "Model Test",
    }
    
    # Build parameters based on model type
    if has_reasoning:
        params = {
            "temperature": 0.2,
            "max_tokens": 2048,
            "reasoning": {"enabled": True}
        }
    else:
        params = {
            "temperature": 0.2,
            "max_tokens": 512
        }
    
    payload = {
        "model": model_id,
        "messages": messages,
        **params
    }
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return result, latency, None
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 45
                print(f"      ⏳ Rate limit (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                error_msg = response.json().get("error", {}).get("message", response.text[:200])
                return None, latency, f"HTTP {response.status_code}: {error_msg}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      ⚠️ Error: {str(e)[:100]}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None, 0, str(e)
    
    return None, 0, "Max retries exceeded"

def create_structured_output(model_id, gsm8k_id, question_text, ground_truth, persona_id, persona_prompt, 
                            api_result, latency, error, has_reasoning=False):
    """Create structured JSON output"""
    success = (api_result is not None and "choices" in api_result)
    content = None
    tokens = {}
    
    if success:
        try:
            content = api_result["choices"][0]["message"].get("content")
        except Exception:
            content = None
        
        # Extract token usage
        usage = api_result.get("usage", {})
        tokens = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens")
        }
        
        # Add reasoning tokens if available
        if has_reasoning and "reasoning_tokens" in usage:
            tokens["reasoning_tokens"] = usage["reasoning_tokens"]
    
    # Build parameters for request
    if has_reasoning:
        request_params = {
            "temperature": 0.2,
            "max_tokens": 2048,
            "model": model_id,
            "reasoning": {"enabled": True}
        }
    else:
        request_params = {
            "temperature": 0.2,
            "max_tokens": 512,
            "model": model_id
        }
    
    structured = {
        "metadata": {
            "run_id": f"{model_id.replace('/', '_')}_man_explicit_001_{gsm8k_id:05d}",
            "experiment_type": "gsm8k_bias_study",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "provider": "openrouter",
                "model_id": model_id,
                "api_endpoint": OPENROUTER_URL,
                "has_reasoning": has_reasoning
            }
        },
        "input": {
            "question_id": gsm8k_id,
            "dataset": "gsm8k_test",
            "question": question_text,
            "ground_truth": ground_truth,
            "persona": {
                "id": persona_id,
                "type": "man_explicit",
                "prompt": persona_prompt
            }
        },
        "request": {
            "messages": [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question_text}
            ],
            "parameters": request_params
        },
        "response": {
            "success": bool(success),
            "content": content,
            "raw_api_response": api_result,
            "error": error,
            "performance": {
                "latency_sec": latency,
                "tokens": tokens
            }
        },
        "analysis": {
            "extracted_answer": None,
            "ground_truth_answer": None,
            "is_correct": None,
            "analysis_timestamp": None
        }
    }
    
    return structured

def process_model(model, personas, gsm8k_data, start_id=1, end_id=10):
    """Process a single model with range of questions"""
    model_id = model["model_id"]
    folder = model["folder"]
    has_reasoning = model["has_reasoning"]
    order = model["order"]
    
    output_dir = f"results/{folder}/man_explicit"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get persona (ID 1)
    persona = next((p for p in personas if p["id"] == 1), None)
    if not persona:
        print(f"  ❌ Persona ID 1 not found!")
        return
    
    print(f"\n[Model {order}] {model_id}")
    print(f"  📁 Output: {output_dir}")
    print(f"  🧠 Reasoning: {has_reasoning}")
    print(f"  📝 Processing questions {start_id}-{end_id}...")
    
    success_count = 0
    error_count = 0
    
    for gsm8k_id in range(start_id, end_id + 1):
        if gsm8k_id > len(gsm8k_data):
            print(f"  ⚠️ Question {gsm8k_id} exceeds available data ({len(gsm8k_data)})")
            break
        
        gsm8k_item = gsm8k_data[gsm8k_id - 1]
        
        # Call API
        messages = [
            {"role": "system", "content": persona["prompt"]},
            {"role": "user", "content": gsm8k_item["question"]}
        ]
        
        api_result, latency, error = call_openrouter_api(model_id, messages, has_reasoning)
        
        # Create structured output
        structured = create_structured_output(
            model_id, gsm8k_id, gsm8k_item["question"], gsm8k_item.get("answer"),
            persona["id"], persona["prompt"], api_result, latency, error, has_reasoning
        )
        
        # Save to file
        output_file = f"{output_dir}/gsm8k_{gsm8k_id:05d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
        
        # Report
        if structured["response"]["success"]:
            print(f"    [{gsm8k_id:5d}] ✅ Success (latency: {latency:.2f}s)")
            success_count += 1
        else:
            print(f"    [{gsm8k_id:5d}] ❌ Error: {error or 'No response'}")
            error_count += 1
        
        # Rate limiting
        time.sleep(3)
    
    # Summary
    total = success_count + error_count
    rate = (success_count / total * 100) if total > 0 else 0
    print(f"  📊 Summary: {success_count}/{total} successful ({rate:.1f}%)")

def main():
    print("🧪 TESTING MODELS 31-35 WITH GSM8K QUESTIONS")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    personas, gsm8k_data = load_data()
    print(f"✅ Loaded {len(personas)} personas and {len(gsm8k_data)} GSM8K questions")
    
    # Process each model
    for model in MODELS:
        try:
            process_model(model, personas, gsm8k_data, start_id=1, end_id=1)
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")

if __name__ == "__main__":
    main()
