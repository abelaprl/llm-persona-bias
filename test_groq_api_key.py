#!/usr/bin/env python3
"""
Test if new API key works with Groq
"""

import json
import requests

API_KEY_NEW = "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"
API_KEY_OLD = "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"

def test_groq_api(api_key, label):
    """Test Groq API"""
    print(f"\n{'='*80}")
    print(f"Testing Groq with: {label}")
    print(f"API Key: {api_key[:20]}...")
    print(f"{'='*80}\n")
    
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
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! API Key works with Groq!")
            content = response.json()['choices'][0]['message']['content']
            print(f"Response: {content}")
            return True
        else:
            print(f"❌ Error: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("\n🔍 TESTING NEW API KEY WITH GROQ\n")
    
    print("Testing OLD API Key:")
    old_result = test_groq_api(API_KEY_OLD, "Old Key (Known Working)")
    
    print("\n" + "="*80)
    
    print("Testing NEW API Key:")
    new_result = test_groq_api(API_KEY_NEW, "New Key")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Old Key: {'✅ Working' if old_result else '❌ Failed'}")
    print(f"New Key: {'✅ Working' if new_result else '❌ Failed'}")
    print("="*80 + "\n")
    
    if new_result:
        print("✨ New API Key is valid for Groq! You can use it for experiments.")
    else:
        print("⚠️  New API Key doesn't work with Groq. Check if it's expired or invalid.")

if __name__ == '__main__':
    main()
