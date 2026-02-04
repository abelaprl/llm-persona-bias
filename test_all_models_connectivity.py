#!/usr/bin/env python3
"""
Test connectivity to all models in config/model_keys.json
Reports which models are actually accessible
"""

import json
import requests
import time
from pathlib import Path

def load_models():
    """Load all models from config"""
    config_path = Path(__file__).parent / "config" / "model_keys.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_groq_model(api_key):
    """Test Groq API"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Say 'OK'"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=10
        )
        if response.status_code == 200:
            return "✅ OK", response.elapsed.total_seconds()
        else:
            return f"❌ {response.status_code}: {response.text[:100]}", None
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}", None

def test_openrouter_model(model_id, api_key):
    """Test OpenRouter API"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say 'OK'"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=10
        )
        if response.status_code == 200:
            return "✅ OK", response.elapsed.total_seconds()
        else:
            error_msg = response.json().get('error', {}).get('message', response.text)
            return f"❌ {response.status_code}: {error_msg[:80]}", None
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}", None

def test_moonshot_model(api_key):
    """Test Moonshot API"""
    try:
        response = requests.post(
            "https://api.moonshot.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": "Say 'OK'"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=10
        )
        if response.status_code == 200:
            return "✅ OK", response.elapsed.total_seconds()
        else:
            return f"❌ {response.status_code}: {response.text[:100]}", None
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}", None

def main():
    """Test all models"""
    models = load_models()
    
    print("\n" + "="*80)
    print("🔌 TESTING ALL MODELS CONNECTIVITY")
    print("="*80 + "\n")
    
    working_models = []
    failed_models = []
    
    for model in models:
        order = model.get('order')
        name = model.get('name', 'Unknown')
        model_id = model.get('model_id', '')
        api_key = model.get('api_key_alias', '')
        provider = model.get('provider', 'openrouter')  # default to openrouter
        
        print(f"[{order:02d}] {name}")
        print(f"     Model ID: {model_id}")
        
        # Determine which API to test
        if 'groq' in model_id.lower():
            status, latency = test_groq_model(api_key)
            print(f"     Provider: Groq")
        elif model.get('provider') == 'moonshot':
            status, latency = test_moonshot_model(api_key)
            print(f"     Provider: Moonshot")
        else:
            status, latency = test_openrouter_model(model_id, api_key)
            print(f"     Provider: OpenRouter")
        
        print(f"     Status: {status}")
        if latency:
            print(f"     Latency: {latency:.2f}s")
        
        if "✅" in status:
            working_models.append((order, name, model_id))
        else:
            failed_models.append((order, name, model_id, status))
        
        print()
        time.sleep(1)  # Rate limiting
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80 + "\n")
    
    print(f"✅ WORKING MODELS ({len(working_models)}):")
    for order, name, model_id in working_models:
        print(f"   [{order:02d}] {name}")
        print(f"        {model_id}")
    
    print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
    for order, name, model_id, status in failed_models:
        print(f"   [{order:02d}] {name}")
        print(f"        {model_id}")
        print(f"        {status}")
    
    print("\n" + "="*80)
    print(f"Total: {len(working_models)}/{len(models)} models working")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
