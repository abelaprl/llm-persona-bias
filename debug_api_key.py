#!/usr/bin/env python3
"""
Debug: Test API key validity with different approaches
"""

import json
import requests
import time

API_KEY = "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"

def test_openrouter_direct():
    """Test OpenRouter with simple request"""
    print("\n" + "="*80)
    print("Test 1: OpenRouter Direct (simple model)")
    print("="*80 + "\n")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/abelaprl/TugasAkhir"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Say OK"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ API Key is valid!")
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_qwen_model():
    """Test with Qwen model"""
    print("\n" + "="*80)
    print("Test 2: OpenRouter with Qwen model")
    print("="*80 + "\n")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/abelaprl/TugasAkhir"
            },
            json={
                "model": "qwen/qwen-max",
                "messages": [{"role": "user", "content": "Say OK"}],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Qwen model works!")
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_google_vertex_direct():
    """Test Google Vertex API directly"""
    print("\n" + "="*80)
    print("Test 3: Google Vertex API Direct")
    print("="*80 + "\n")
    
    try:
        # Try direct Google Vertex endpoint
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers={
                "Content-Type": "application/json"
            },
            params={
                "key": API_KEY
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": "Say OK"}
                        ]
                    }
                ]
            },
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ API Key works with Google Vertex!")
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_models_list():
    """Get available models list"""
    print("\n" + "="*80)
    print("Test 4: Get Available Models from OpenRouter")
    print("="*80 + "\n")
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()['data']
            print(f"\n✅ Got {len(models)} available models\n")
            
            # Search for relevant models
            keywords = ['qwen', 'gpt-oss', 'llama-3.3']
            for keyword in keywords:
                matching = [m for m in models if keyword.lower() in m['id'].lower()]
                if matching:
                    print(f"\n🔍 Models matching '{keyword}':")
                    for m in matching[:3]:  # Show first 3
                        print(f"   - {m['id']}")
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    print("\n🔍 DEBUGGING API KEY AND MODEL ACCESS\n")
    
    test_openrouter_direct()
    test_qwen_model()
    test_google_vertex_direct()
    test_models_list()
    
    print("\n" + "="*80)
    print("Debug complete")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
