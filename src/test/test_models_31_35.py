#!/usr/bin/env python3
"""
Test models 31-35 connectivity with 1 man_explicit question
"""

import requests
import json
import time

# Models to test
MODELS = [
    {"order": 31, "name": "OpenAI: GPT-OSS-20B", "model_id": "openai/gpt-oss-20b"},
    {"order": 32, "name": "Qwen: Qwen3 Next 80B A3B (Thinking)", "model_id": "qwen/qwen3-next-80b-a3b-thinking"},
    {"order": 33, "name": "Qwen: Qwen3 Next 80B A3B (Instruct)", "model_id": "qwen/qwen3-next-80b-a3b-instruct"},
    {"order": 34, "name": "OpenAI: GPT-OSS-120B", "model_id": "openai/gpt-oss-120b"},
    {"order": 35, "name": "Meta: Llama 3.3 70B Instruct", "model_id": "meta-llama/llama-3.3-70b-instruct"},
]

# Shared API key (Google Vertex via OpenRouter)
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Test question from man_explicit
TEST_QUESTION = "The school soccer team has 4 soccer balls. The school soccer coach ordered 5 more soccer balls. How many soccer balls does the school soccer team have now?"

def test_model(model_id, model_name, order):
    """Test single model with timeout"""
    print(f"\n[{order}] Testing: {model_name}")
    print(f"    Model ID: {model_id}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "https://github.com/user/project",
                "X-Title": "Model Test",
            },
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful math problem solver."
                    },
                    {
                        "role": "user",
                        "content": TEST_QUESTION
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'][:100]
                print(f"    ✅ SUCCESS (latency: {elapsed:.2f}s)")
                print(f"    Response: {content}...")
                return True
            else:
                print(f"    ❌ FAILED: No choices in response")
                print(f"    Response: {result}")
                return False
        else:
            error_msg = response.json().get('error', {})
            error_type = error_msg.get('type', 'Unknown')
            error_message = error_msg.get('message', 'No message')
            print(f"    ❌ ERROR {response.status_code}: {error_type}")
            print(f"    Message: {error_message}")
            return False
            
    except requests.Timeout:
        print(f"    ❌ TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"    ❌ EXCEPTION: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("Testing Models 31-35 Connectivity")
    print("="*80)
    
    results = {}
    
    for model in MODELS:
        success = test_model(model['model_id'], model['name'], model['order'])
        results[f"[{model['order']}] {model['name']}"] = "✅ PASS" if success else "❌ FAIL"
        time.sleep(2)  # Rate limiting
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for model_name, status in results.items():
        print(f"{status} - {model_name}")
    
    passed = sum(1 for v in results.values() if "PASS" in v)
    total = len(results)
    print(f"\nTotal: {passed}/{total} models working")

if __name__ == "__main__":
    main()
