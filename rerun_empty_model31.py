"""Re-run empty responses for model 31 (allenai_olmo-3-32b-think) with higher max_tokens"""
import json
import os
import time
import requests
from datetime import datetime

# OpenRouter API key (from config - using different key)
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "allenai/olmo-3-32b-think"
MODEL_DIR = "result/allenai_olmo-3-32b-think"

# Higher max_tokens to avoid truncation (thinking models need more)
NEW_MAX_TOKENS = 8192  # 4x the original

def call_api(messages, max_tokens=NEW_MAX_TOKENS, temperature=0.2):
    """Call OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "GSM8K Experiment"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        return response.json(), latency
    else:
        print(f"API Error {response.status_code}: {response.text[:200]}")
        return None, latency

def is_empty_response(data):
    """Check if response is empty (content is empty string)"""
    content = data.get('response', {}).get('content', '')
    return not content or content.strip() == ''

def process_empty_files():
    """Find and re-run all empty response files"""
    personas = [
        'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
        'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
    ]
    
    empty_files = []
    
    # First pass: find all empty files
    print("Scanning for empty response files...")
    for persona in personas:
        persona_path = os.path.join(MODEL_DIR, persona)
        if not os.path.exists(persona_path):
            continue
        
        files = sorted([f for f in os.listdir(persona_path) if f.endswith('.json')])
        for fname in files:
            fpath = os.path.join(persona_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if is_empty_response(data):
                    empty_files.append((persona, fname, fpath, data))
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
    
    print(f"\nFound {len(empty_files)} empty response files to re-run")
    
    if not empty_files:
        print("No empty files to process!")
        return
    
    # Second pass: re-run each empty file
    success_count = 0
    fail_count = 0
    
    for i, (persona, fname, fpath, original_data) in enumerate(empty_files):
        print(f"\n[{i+1}/{len(empty_files)}] Processing {persona}/{fname}...")
        
        # Get original request messages
        messages = original_data.get('request', {}).get('messages', [])
        if not messages:
            print(f"  SKIP: No messages found in {fpath}")
            fail_count += 1
            continue
        
        # Call API with higher max_tokens
        api_response, latency = call_api(messages)
        
        if api_response is None:
            print(f"  FAIL: API call failed")
            fail_count += 1
            time.sleep(2)  # Wait before retry
            continue
        
        # Extract new content
        choices = api_response.get('choices', [])
        if not choices:
            print(f"  FAIL: No choices in response")
            fail_count += 1
            continue
        
        new_content = choices[0].get('message', {}).get('content', '')
        finish_reason = choices[0].get('finish_reason', '')
        
        print(f"  Content length: {len(new_content)}, finish_reason: {finish_reason}")
        
        # Update the JSON file
        original_data['response']['content'] = new_content
        original_data['response']['raw_api_response'] = api_response
        original_data['response']['latency'] = latency
        original_data['request']['parameters']['max_tokens'] = NEW_MAX_TOKENS
        original_data['metadata']['rerun_timestamp'] = datetime.now().isoformat()
        original_data['metadata']['rerun_reason'] = 'empty_response_fix'
        
        # Save back
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)
            print(f"  SUCCESS: Updated {fname}")
            success_count += 1
        except Exception as e:
            print(f"  FAIL: Could not save: {e}")
            fail_count += 1
        
        # Rate limiting - wait between requests
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED: {success_count} success, {fail_count} failed")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Auto-run without confirmation
    print("="*60)
    print("RE-RUN EMPTY RESPONSES FOR MODEL 31")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"New max_tokens: {NEW_MAX_TOKENS}")
    print(f"API endpoint: {API_URL}")
    print()
    
    process_empty_files()
