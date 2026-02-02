#!/usr/bin/env python3
"""
Script untuk mengecek semua model API yang tersedia dari config
"""

import json
import os
from collections import defaultdict

def load_model_config():
    """Load model configuration from config/model_keys.json"""
    config_path = os.path.join("config", "model_keys.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file tidak ditemukan: {config_path}")
        return None
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def analyze_models(config):
    """Analyze and categorize available models"""
    if not config:
        return
    
    # Group by provider
    providers = defaultdict(list)
    
    print("🤖 DAFTAR MODEL API YANG TERSEDIA")
    print("=" * 60)
    print(f"Total model: {len(config)}")
    print()
    
    # Display all models with details
    for model in sorted(config, key=lambda x: x.get('order', 0)):
        order = model.get('order', 'N/A')
        name = model.get('name', 'Unknown')
        model_id = model.get('model_id', 'Unknown')
        provider = model.get('provider', 'openrouter')  # default to openrouter
        
        # Extract provider from model_id if not specified
        if provider == 'openrouter' and '/' in model_id:
            extracted_provider = model_id.split('/')[0]
            providers[extracted_provider].append(model)
        else:
            providers[provider].append(model)
        
        print(f"[{order:2}] {name}")
        print(f"    Model ID: {model_id}")
        if 'endpoint_url' in model:
            print(f"    Endpoint: {model['endpoint_url']}")
        print(f"    Provider: {provider}")
        print()
    
    # Summary by provider
    print("\n📊 RINGKASAN PER PROVIDER")
    print("=" * 40)
    
    for provider, models in sorted(providers.items()):
        print(f"\n🔹 {provider.upper()} ({len(models)} models):")
        for model in models:
            model_id = model.get('model_id', 'Unknown')
            is_free = ':free' in model_id
            status = "🆓 FREE" if is_free else "💰 PAID"
            print(f"   • {model.get('name', 'Unknown')} {status}")
    
    # Count free vs paid models
    free_count = sum(1 for model in config if ':free' in model.get('model_id', ''))
    paid_count = len(config) - free_count
    
    print(f"\n📈 STATISTIK:")
    print(f"   Total Models: {len(config)}")
    print(f"   Free Models: {free_count}")
    print(f"   Paid Models: {paid_count}")

def get_model_by_order(config, order):
    """Get model by order number"""
    for model in config:
        if model.get('order') == order:
            return model
    return None

def list_free_models(config):
    """List only free models"""
    print("\n🆓 MODEL GRATIS YANG TERSEDIA:")
    print("=" * 50)
    
    free_models = [model for model in config if ':free' in model.get('model_id', '')]
    
    for model in sorted(free_models, key=lambda x: x.get('order', 0)):
        order = model.get('order', 'N/A')
        name = model.get('name', 'Unknown')
        model_id = model.get('model_id', 'Unknown')
        print(f"[{order:2}] {name}")
        print(f"    {model_id}")
        print()

def main():
    """Main function"""
    config = load_model_config()
    
    if not config:
        return
    
    # Analyze all models
    analyze_models(config)
    
    # List free models separately
    list_free_models(config)
    
    print("\n✅ Analisis selesai!")
    print("\nCara penggunaan:")
    print("1. Gunakan 'order' number untuk memilih model")
    print("2. Atau gunakan 'model_id' langsung")
    print("3. Model dengan ':free' tidak memerlukan pembayaran")

if __name__ == "__main__":
    main()