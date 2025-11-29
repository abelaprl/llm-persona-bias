"""
Test script to verify connectivity to all available models (excluding Nemotron).
Sends a simple test question to each model and logs success/failure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import requests
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_keys import load_model_keys, get_models


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"


def call_openrouter(model_id: str, api_key: str, test_prompt: str) -> tuple[bool, str, float]:
    """Call OpenRouter API with test prompt."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": test_prompt},
        ],
        "max_tokens": 100,  # Keep response short for testing
    }
    
    start = time.time()
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        latency = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and len(result["choices"]) > 0:
                return True, "SUCCESS", latency
            else:
                return False, f"No choices in response: {result}", latency
        else:
            error_msg = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return False, f"HTTP {resp.status_code}: {error_msg}", latency
    except requests.Timeout:
        latency = time.time() - start
        return False, "TIMEOUT (30s)", latency
    except Exception as e:
        latency = time.time() - start
        return False, f"ERROR: {str(e)[:100]}", latency


def call_moonshot(model_id: str, api_key: str, test_prompt: str) -> tuple[bool, str, float]:
    """Call Moonshot API with test prompt."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": test_prompt},
        ],
        "max_tokens": 100,
    }
    
    start = time.time()
    try:
        resp = requests.post(MOONSHOT_URL, headers=headers, json=payload, timeout=30)
        latency = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and len(result["choices"]) > 0:
                return True, "SUCCESS", latency
            else:
                return False, f"No choices in response: {result}", latency
        else:
            error_msg = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return False, f"HTTP {resp.status_code}: {error_msg}", latency
    except requests.Timeout:
        latency = time.time() - start
        return False, "TIMEOUT (30s)", latency
    except Exception as e:
        latency = time.time() - start
        return False, f"ERROR: {str(e)[:100]}", latency


def main():
    print("=" * 80)
    print("MODEL CONNECTIVITY TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load models
    try:
        keys = load_model_keys()
        models = get_models(keys)
    except Exception as e:
        print(f"Error loading model keys: {e}")
        return
    
    # Test prompt (simple question)
    test_prompt = "What is 2 + 2? Answer briefly in one line."
    
    results = []
    
    # Filter out Nemotron (order 3) and test each model
    for model in models:
        order = model.get("order")
        name = model.get("name", "Unknown")
        model_id = model.get("model_id")
        api_key = model.get("api_key_alias")
        provider = model.get("provider", "openrouter")  # Default to openrouter
        
        # Skip Nemotron
        if order == 3:
            print(f"[SKIP] Order {order}: {name} (Nemotron - excluded)")
            continue
        
        # Skip if no API key
        if not api_key:
            print(f"[SKIP] Order {order}: {name} - No API key")
            results.append({
                "order": order,
                "name": name,
                "model_id": model_id,
                "status": "SKIPPED",
                "message": "No API key",
                "latency": None
            })
            continue
        
        print(f"[TEST] Order {order}: {name}")
        print(f"       Model ID: {model_id}")
        print(f"       Provider: {provider}", end="")
        
        # Call appropriate API based on provider
        if provider == "moonshot":
            success, message, latency = call_moonshot(model_id, api_key, test_prompt)
        else:  # Default to openrouter
            success, message, latency = call_openrouter(model_id, api_key, test_prompt)
        
        status = "✓ OK" if success else "✗ FAIL"
        print(f" ... {status}")
        
        if success:
            print(f"       Latency: {latency:.2f}s")
        else:
            print(f"       Error: {message}")
        
        results.append({
            "order": order,
            "name": name,
            "model_id": model_id,
            "provider": provider,
            "status": "SUCCESS" if success else "FAILED",
            "message": message if not success else "Connected successfully",
            "latency": round(latency, 2) if success else None
        })
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] == "FAILED"]
    skipped = [r for r in results if r["status"] == "SKIPPED"]
    
    print(f"Total tested: {len(successful) + len(failed)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"  - Skipped: {len(skipped)}")
    
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['name']}: {r['message']}")
    
    # Save results to file
    log_file = Path("logs_model_connectivity_test.json")
    with log_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {log_file}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
