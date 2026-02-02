#!/usr/bin/env python3
"""
Script untuk menguji konektivitas dan status semua model API gratis
"""

import json
import os
import time
import requests
from datetime import datetime
import concurrent.futures
from threading import Lock

# Global variables
results_lock = Lock()
test_results = []

def load_free_models():
    """Load only free models from configuration"""
    config_path = os.path.join("config", "model_keys.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file tidak ditemukan: {config_path}")
        return []
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            all_models = json.load(f)
        
        # Filter hanya model gratis
        free_models = []
        for model in all_models:
            model_id = model.get("model_id", "")
            # Skip model berbayar
            if (":free" in model_id or 
                model_id in ["deepseek/deepseek-r1:free"] or 
                "free" in model.get("name", "").lower()):
                free_models.append(model)
        
        return free_models
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return []

def test_openrouter_model(model_info):
    """Test single OpenRouter model"""
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {model_info['api_key_alias']}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_info["model_id"],
        "messages": [{"role": "user", "content": "Hello, respond with just 'OK'"}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "status": "✅ SUCCESS",
                    "response": content.strip()[:100],
                    "latency": f"{latency:.2f}s",
                    "usage": usage,
                    "error": None
                }
        
        return {
            "status": "❌ FAILED",
            "response": None,
            "latency": f"{latency:.2f}s",
            "usage": None,
            "error": f"HTTP {response.status_code}: {response.text[:200]}"
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "⏰ TIMEOUT",
            "response": None,
            "latency": "30s+",
            "usage": None,
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "status": "💥 ERROR",
            "response": None,
            "latency": "N/A",
            "usage": None,
            "error": str(e)
        }

def test_moonshot_model(model_info):
    """Test Moonshot model with custom endpoint"""
    endpoint = model_info.get("endpoint_url", "https://api.moonshot.ai/v1/chat/completions")
    
    headers = {
        "Authorization": f"Bearer {model_info['api_key_alias']}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_info["model_id"],
        "messages": [{"role": "user", "content": "Hello, respond with just 'OK'"}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        start_time = time.time()
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "status": "✅ SUCCESS",
                    "response": content.strip()[:100],
                    "latency": f"{latency:.2f}s",
                    "usage": usage,
                    "error": None
                }
        
        return {
            "status": "❌ FAILED",
            "response": None,
            "latency": f"{latency:.2f}s",
            "usage": None,
            "error": f"HTTP {response.status_code}: {response.text[:200]}"
        }
        
    except Exception as e:
        return {
            "status": "💥 ERROR",
            "response": None,
            "latency": "N/A",
            "usage": None,
            "error": str(e)
        }

def test_single_model(model_info):
    """Test a single model and return results"""
    order = model_info.get("order", "?")
    name = model_info.get("name", "Unknown")
    model_id = model_info.get("model_id", "Unknown")
    
    print(f"🧪 Testing [{order}] {name}...")
    
    # Determine which test function to use
    if model_info.get("provider") == "moonshot":
        result = test_moonshot_model(model_info)
    else:
        result = test_openrouter_model(model_info)
    
    # Prepare result data
    test_result = {
        "order": order,
        "name": name,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        **result
    }
    
    # Thread-safe result storage
    with results_lock:
        test_results.append(test_result)
    
    # Print immediate result
    status_color = "🟢" if "SUCCESS" in result["status"] else "🔴"
    print(f"  {status_color} [{order}] {name}: {result['status']} ({result['latency']})")
    
    return test_result

def run_connectivity_tests():
    """Run connectivity tests for all free models"""
    free_models = load_free_models()
    
    if not free_models:
        print("❌ Tidak ada model gratis yang ditemukan!")
        return
    
    print(f"🚀 TESTING {len(free_models)} MODEL GRATIS")
    print("=" * 60)
    print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test models concurrently (but limit concurrent requests)
    max_workers = 5  # Limit concurrent API calls
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_single_model, model) for model in free_models]
        
        # Wait for all tests to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error in test: {e}")
    
    # Sort results by order
    test_results.sort(key=lambda x: x["order"])
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 HASIL TEST KONEKTIVITAS MODEL")
    print("=" * 80)
    
    working_models = []
    failed_models = []
    
    for result in test_results:
        status = result["status"]
        order = result["order"]
        name = result["name"]
        latency = result["latency"]
        error = result.get("error")
        
        if "SUCCESS" in status:
            working_models.append(result)
            print(f"✅ [{order:2}] {name}")
            print(f"    Response: {result.get('response', 'N/A')}")
            print(f"    Latency: {latency}")
            if result.get('usage'):
                print(f"    Usage: {result['usage']}")
        else:
            failed_models.append(result)
            print(f"❌ [{order:2}] {name}")
            print(f"    Status: {status}")
            print(f"    Error: {error}")
        print()
    
    # Summary statistics
    print("📈 RINGKASAN:")
    print(f"  ✅ Model yang berfungsi: {len(working_models)}")
    print(f"  ❌ Model yang gagal: {len(failed_models)}")
    print(f"  📊 Success rate: {len(working_models)/len(test_results)*100:.1f}%")
    
    # Save detailed results
    save_results_to_file()
    
    # Show recommended models
    if working_models:
        print("\n🌟 MODEL YANG DIREKOMENDASIKAN:")
        print("-" * 40)
        # Sort by latency (fastest first)
        working_models.sort(key=lambda x: float(x["latency"].replace("s", "")))
        
        for i, model in enumerate(working_models[:10], 1):  # Top 10
            print(f"{i:2}. [{model['order']:2}] {model['name']}")
            print(f"    Model ID: {model['model_id']}")
            print(f"    Latency: {model['latency']}")
            print()

def save_results_to_file():
    """Save test results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs_model_connectivity_test.json"
    
    output_data = {
        "test_timestamp": datetime.now().isoformat(),
        "total_tested": len(test_results),
        "results": test_results
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Hasil test disimpan ke: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    run_connectivity_tests()