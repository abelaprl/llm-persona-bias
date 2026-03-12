#!/usr/bin/env python3
"""
Test script to check accessibility of models 31-42 with the provided API key
"""

import json
import sys
import time
from pathlib import Path

# Models to test (order 31-42)
models_to_test = [
    {"order": 31, "name": "OpenAI: GPT-OSS-20B", "model_id": "openai/gpt-oss-20b"},
    {"order": 32, "name": "Qwen: Qwen3 Next 80B A3B (Thinking)", "model_id": "qwen/qwen3-next-80b-a3b-thinking"},
    {"order": 33, "name": "Qwen: Qwen3 235B A22B Instruct 2507", "model_id": "qwen/qwen3-235b-a22b-2507"},
    {"order": 34, "name": "OpenAI: GPT-OSS-120B", "model_id": "openai/gpt-oss-120b"},
    {"order": 35, "name": "Meta: Llama 3.3 70B Instruct", "model_id": "meta-llama/llama-3.3-70b-instruct"},
    {"order": 36, "name": "Google: Gemma 3 27B", "model_id": "google/gemma-3-27b-it"},
    {"order": 37, "name": "Google: Gemma 2 9B", "model_id": "google/gemma-2-9b-it"},
    {"order": 38, "name": "DeepSeek V3.2", "model_id": "deepseek/deepseek-v3.2"},
    {"order": 39, "name": "Google: Gemini 2.5 Flash", "model_id": "google/gemini-2.5-flash"},
    {"order": 40, "name": "Meta: Llama 3.1 8B Instruct", "model_id": "meta-llama/llama-3.1-8b-instruct"},
    {"order": 41, "name": "Mistral: Ministral 8B", "model_id": "mistralai/ministral-8b-2512"},
    {"order": 42, "name": "Allen AI: OLMo 3 32B Think", "model_id": "allenai/olmo-3-32b-think"}
]

# API configuration
NEW_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def test_model_access(model_info, api_key):
    """
    Test if a model can be accessed with the given API key
    """
    try:
        import requests
        
        model_name = model_info["name"]
        model_id = model_info["model_id"]
        order = model_info["order"]
        
        print(f"\n[{order}] Testing {model_name} ({model_id})...")
        
        # Make a simple request with a short prompt and low max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Test Script"
        }
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✓ Model {order} ({model_name}): ACCESSIBLE")
            return {"order": order, "name": model_name, "model_id": model_id, "status": "✓ ACCESSIBLE", "code": 200}
        elif response.status_code == 401:
            print(f"✗ Model {order} ({model_name}): UNAUTHORIZED (API key issue)")
            return {"order": order, "name": model_name, "model_id": model_id, "status": "✗ UNAUTHORIZED", "code": 401}
        elif response.status_code == 404:
            print(f"✗ Model {order} ({model_name}): NOT FOUND")
            return {"order": order, "name": model_name, "model_id": model_id, "status": "✗ NOT FOUND", "code": 404}
        else:
            error_msg = response.text if response.text else str(response.status_code)
            print(f"✗ Model {order} ({model_name}): ERROR {response.status_code}")
            print(f"   Response: {error_msg[:200]}")
            return {"order": order, "name": model_name, "model_id": model_id, "status": f"✗ ERROR {response.status_code}", "code": response.status_code}
    
    except requests.exceptions.Timeout:
        print(f"✗ Model {order}: TIMEOUT")
        return {"order": order, "name": model_name, "model_id": model_id, "status": "✗ TIMEOUT", "code": None}
    except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
        print(f"✗ Model {order}: CONNECTION ERROR")
        return {"order": order, "name": model_name, "model_id": model_id, "status": "✗ CONNECTION ERROR", "code": None}
    except KeyboardInterrupt:
        print(f"\n✗ Model {order}: INTERRUPTED BY USER")
        raise
    except Exception as e:
        print(f"✗ Model {order}: ERROR - {str(e)[:100]}")
        return {"order": order, "name": model_name, "model_id": model_id, "status": f"✗ ERROR: {str(e)[:50]}", "code": None}

def main():
    print("=" * 80)
    print("Testing Model Accessibility with New API Key")
    print("=" * 80)
    print(f"API Key: {NEW_API_KEY[:20]}...{NEW_API_KEY[-10:]}")
    print(f"Testing {len(models_to_test)} models (orders 31-42)")
    print("=" * 80)
    
    results = []
    
    for model_info in models_to_test:
        result = test_model_access(model_info, NEW_API_KEY)
        results.append(result)
        time.sleep(2)  # Longer delay between requests to avoid rate limiting
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    accessible = [r for r in results if "ACCESSIBLE" in r["status"]]
    inaccessible = [r for r in results if "ACCESSIBLE" not in r["status"]]
    
    print(f"\nAccessible Models: {len(accessible)}/{len(results)}")
    for r in accessible:
        print(f"  [{r['order']}] {r['name']}")
    
    print(f"\nInaccessible Models: {len(inaccessible)}/{len(results)}")
    for r in inaccessible:
        print(f"  [{r['order']}] {r['name']}: {r['status']}")
    
    # Save results to JSON
    output_file = Path("test_api_key_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "api_key": NEW_API_KEY[:20] + "..." + NEW_API_KEY[-10:],
            "test_date": str(time.strftime("%Y-%m-%d %H:%M:%S")),
            "total_models": len(results),
            "accessible_count": len(accessible),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: requests library not found. Please install it with: pip install requests")
        sys.exit(1)
    
    main()
