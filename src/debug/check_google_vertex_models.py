#!/usr/bin/env python3
"""
Check available models from Google Vertex provider on OpenRouter
Filter for cheap/efficient models
"""

import requests
import json

API_KEY = "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"

def get_all_models():
    """Get all available models"""
    print("\n" + "="*80)
    print("📋 FETCHING ALL AVAILABLE MODELS FROM OPENROUTER")
    print("="*80 + "\n")
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15
        )
        
        if response.status_code == 200:
            models = response.json()['data']
            print(f"✅ Got {len(models)} total models\n")
            return models
        else:
            print(f"❌ Error: {response.json()}")
            return []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def filter_cheap_models(models):
    """Filter models by criteria: price, inference speed, capability"""
    print("\n" + "="*80)
    print("🔍 FILTERING MODELS BY PRICE & CAPABILITIES")
    print("="*80 + "\n")
    
    # Keywords untuk cari model murah/bagus
    keywords = {
        'llama': 'Meta Llama (open source)',
        'deepseek': 'DeepSeek (cost-effective)',
        'qwen': 'Qwen (Chinese, cheap)',
        'gpt-4o-mini': 'OpenAI small (fast)',
        'mistral': 'Mistral (efficient)',
        'gemini': 'Google Gemini',
        'claude': 'Anthropic Claude'
    }
    
    filtered = {}
    
    for keyword, description in keywords.items():
        matching = [m for m in models if keyword.lower() in m['id'].lower()]
        if matching:
            filtered[description] = matching[:3]  # Top 3 per category
    
    return filtered

def test_model(model_id):
    """Test single model"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/abelaprl/TugasAkhir"
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say OK"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=15
        )
        
        if response.status_code == 200:
            latency = response.elapsed.total_seconds()
            return {"success": True, "latency": latency}
        else:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            return {"success": False, "error": str(error)[:80]}
    except Exception as e:
        return {"success": False, "error": str(e)[:80]}

def main():
    """Main function"""
    # Get all models
    models = get_all_models()
    
    if not models:
        print("Failed to fetch models")
        return
    
    # Filter cheap models
    filtered = filter_cheap_models(models)
    
    print("📊 FILTERED MODELS:\n")
    for category, model_list in filtered.items():
        print(f"\n{category}:")
        print("-" * 80)
        for m in model_list:
            print(f"  ID: {m['id']}")
            print(f"     Name: {m.get('name', 'N/A')}")
            
            # Price info
            pricing = m.get('pricing', {})
            prompt_price = pricing.get('prompt', 'N/A')
            completion_price = pricing.get('completion', 'N/A')
            print(f"     Pricing: ${prompt_price}/1k tokens (prompt) | ${completion_price}/1k tokens (completion)")
            
            # Capabilities
            caps = m.get('architecture', {})
            print(f"     Context: {m.get('context_length', 'Unknown')} tokens")
            print()
    
    # Test a few promising models
    print("\n" + "="*80)
    print("🧪 TESTING SAMPLE MODELS")
    print("="*80 + "\n")
    
    test_candidates = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat",
        "qwen/qwen-max-latest",
        "openai/gpt-4o-mini",
        "mistralai/mistral-large-2411"
    ]
    
    results = []
    for model_id in test_candidates:
        print(f"Testing: {model_id}")
        result = test_model(model_id)
        
        if result["success"]:
            print(f"  ✅ Success (latency: {result['latency']:.2f}s)")
            results.append((model_id, True, result['latency']))
        else:
            print(f"  ❌ Failed: {result['error']}")
            results.append((model_id, False, None))
        print()
    
    # Summary
    print("\n" + "="*80)
    print("✅ WORKING MODELS")
    print("="*80 + "\n")
    
    working = [r for r in results if r[1]]
    if working:
        for model_id, _, latency in working:
            print(f"✅ {model_id} (latency: {latency:.2f}s)")
    else:
        print("No models tested successfully")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
