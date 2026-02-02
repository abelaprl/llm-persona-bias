#!/usr/bin/env python3
"""
Test khusus untuk Groq Llama model
"""

import json
import os
import time
import requests
from datetime import datetime

def test_groq_model():
    """Test Groq Llama 3.1 8B Instant model specifically"""
    
    # Load config to get Groq model info
    config_path = os.path.join("config", "model_keys.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file tidak ditemukan: {config_path}")
        return
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            models = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Find Groq model
    groq_model = None
    for model in models:
        if "groq" in model.get("model_id", "").lower():
            groq_model = model
            break
    
    if not groq_model:
        print("❌ Groq model tidak ditemukan dalam config")
        return
    
    print("🧪 TESTING GROQ LLAMA MODEL")
    print("=" * 50)
    print(f"Model: {groq_model['name']}")
    print(f"Model ID: {groq_model['model_id']}")
    print(f"API Key: {groq_model['api_key_alias'][:8]}...")
    print()
    
    # Test dengan endpoint yang berbeda-beda
    endpoints_to_test = [
        {
            "name": "OpenRouter (Default)",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {groq_model['api_key_alias']}",
                "Content-Type": "application/json",
            }
        },
        {
            "name": "Direct Groq API", 
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "headers": {
                "Authorization": f"Bearer {groq_model['api_key_alias']}",
                "Content-Type": "application/json",
            }
        }
    ]
    
    test_payload = {
        "model": groq_model["model_id"],
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just 'Groq is working!' to test connectivity."}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    for endpoint in endpoints_to_test:
        print(f"🔍 Testing endpoint: {endpoint['name']}")
        print(f"   URL: {endpoint['url']}")
        
        try:
            start_time = time.time()
            response = requests.post(
                endpoint['url'], 
                headers=endpoint['headers'], 
                json=test_payload, 
                timeout=30
            )
            latency = time.time() - start_time
            
            print(f"   ⏱️ Latency: {latency:.2f}s")
            print(f"   📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        
                        print(f"   ✅ SUCCESS!")
                        print(f"   🤖 Response: {content}")
                        print(f"   📈 Usage: {usage}")
                        print(f"   📋 Full response keys: {list(data.keys())}")
                        
                        return True, endpoint['name'], content, latency, usage
                    else:
                        print(f"   ❌ No choices in response: {data}")
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response")
            else:
                print(f"   ❌ FAILED: {response.status_code}")
                print(f"   📄 Response: {response.text[:300]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ TIMEOUT (30s)")
        except Exception as e:
            print(f"   💥 ERROR: {e}")
        
        print()
    
    return False, None, None, None, None

def test_alternative_groq_keys():
    """Test if we can use the Groq key directly with Groq API"""
    
    # Load config
    config_path = os.path.join("config", "model_keys.json")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            models = json.load(f)
    except:
        return
    
    # Find Groq model
    groq_model = None
    for model in models:
        if "groq" in model.get("model_id", "").lower():
            groq_model = model
            break
    
    if not groq_model:
        return
    
    print("🔄 TESTING ALTERNATIF GROQ CONFIGURATIONS")
    print("=" * 60)
    
    # Test different model names that might work with Groq API
    alternative_models = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192", 
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
    
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_model['api_key_alias']}",
        "Content-Type": "application/json",
    }
    
    for model_name in alternative_models:
        print(f"🧪 Testing model: {model_name}")
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Say 'Hello from Groq!' if this works."}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        try:
            start_time = time.time()
            response = requests.post(groq_api_url, headers=headers, json=payload, timeout=15)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    print(f"   ✅ SUCCESS! ({latency:.2f}s)")
                    print(f"   🤖 Response: {content}")
                    print(f"   📊 Usage: {usage}")
                    
                    return True, model_name, content, latency
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        print()
    
    return False, None, None, None

if __name__ == "__main__":
    print("🚀 GROQ LLAMA CONNECTIVITY TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test standard configuration
    success, endpoint, response, latency, usage = test_groq_model()
    
    if not success:
        print("🔄 Standard test failed, trying alternative configurations...")
        print()
        alt_success, alt_model, alt_response, alt_latency = test_alternative_groq_keys()
        
        if alt_success:
            print("🎉 GROQ ALTERNATIVE CONFIGURATION WORKS!")
            print(f"   Working model: {alt_model}")
            print(f"   Latency: {alt_latency:.2f}s")
        else:
            print("😞 All Groq tests failed")
    else:
        print("🎉 GROQ MODEL WORKS!")
        print(f"   Working endpoint: {endpoint}")
        print(f"   Latency: {latency:.2f}s")
    
    print("\n" + "=" * 60)
    print("Test completed!")