#!/usr/bin/env python3
"""
List ALL available models from Google Vertex with full details
Export to CSV for easy analysis
"""

import requests
import json
import csv
from pathlib import Path

API_KEY = "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"

def get_all_models():
    """Fetch all available models"""
    print("\n" + "="*100)
    print("📋 FETCHING ALL AVAILABLE MODELS FROM GOOGLE VERTEX (OpenRouter)")
    print("="*100 + "\n")
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30
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

def save_models_to_csv(models):
    """Save models to CSV"""
    csv_path = Path("available_models_google_vertex.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model ID', 
            'Name', 
            'Prompt Price ($/1k tokens)',
            'Completion Price ($/1k tokens)',
            'Context Length',
            'Provider',
            'Architecture'
        ])
        
        for model in models:
            pricing = model.get('pricing', {})
            prompt_price = pricing.get('prompt', 'N/A')
            completion_price = pricing.get('completion', 'N/A')
            context = model.get('context_length', 'Unknown')
            provider = model.get('owner', 'Unknown')
            arch = model.get('architecture', {}).get('modality', 'Unknown')
            
            writer.writerow([
                model['id'],
                model.get('name', 'N/A'),
                prompt_price,
                completion_price,
                context,
                provider,
                arch
            ])
    
    print(f"📊 Saved to: {csv_path}\n")

def print_models_by_category(models):
    """Print models organized by category"""
    categories = {}
    
    for model in models:
        # Categorize by provider/name
        name = model.get('name', '')
        model_id = model['id']
        
        # Extract category from name or id
        if 'meta' in model_id.lower() or 'llama' in name.lower():
            cat = 'Meta Llama'
        elif 'deepseek' in model_id.lower():
            cat = 'DeepSeek'
        elif 'qwen' in model_id.lower():
            cat = 'Qwen'
        elif 'mistral' in model_id.lower():
            cat = 'Mistral'
        elif 'openai' in model_id.lower():
            cat = 'OpenAI'
        elif 'google' in model_id.lower() or 'gemini' in name.lower():
            cat = 'Google Gemini'
        elif 'claude' in model_id.lower():
            cat = 'Anthropic Claude'
        elif 'grok' in model_id.lower():
            cat = 'xAI Grok'
        else:
            cat = 'Other'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(model)
    
    # Print by category
    for cat in sorted(categories.keys()):
        models_in_cat = categories[cat]
        print(f"\n{'='*100}")
        print(f"🏷️  {cat} ({len(models_in_cat)} models)")
        print(f"{'='*100}")
        
        # Sort by price (prompt)
        sorted_models = sorted(
            models_in_cat, 
            key=lambda m: float(m.get('pricing', {}).get('prompt', 999))
        )
        
        for i, model in enumerate(sorted_models[:10], 1):  # Show top 10 per category
            pricing = model.get('pricing', {})
            prompt_price = pricing.get('prompt', 'N/A')
            completion_price = pricing.get('completion', 'N/A')
            context = model.get('context_length', 'Unknown')
            
            print(f"\n[{i}] {model['id']}")
            print(f"    Name: {model.get('name', 'N/A')}")
            print(f"    Pricing: ${prompt_price}/1k (prompt) | ${completion_price}/1k (completion)")
            print(f"    Context: {context} tokens")

def print_cheapest_models(models, count=15):
    """Print cheapest models"""
    print(f"\n\n{'='*100}")
    print(f"💰 TOP {count} CHEAPEST MODELS (by prompt price)")
    print(f"{'='*100}\n")
    
    # Filter models with valid pricing
    priced_models = [
        m for m in models 
        if isinstance(m.get('pricing', {}).get('prompt'), (int, float))
    ]
    
    # Sort by prompt price
    sorted_models = sorted(
        priced_models,
        key=lambda m: float(m.get('pricing', {}).get('prompt', 999))
    )
    
    for i, model in enumerate(sorted_models[:count], 1):
        pricing = model.get('pricing', {})
        prompt_price = pricing.get('prompt', 'N/A')
        completion_price = pricing.get('completion', 'N/A')
        
        print(f"[{i:2d}] ${float(prompt_price):.10f} | {model['id']:<50}")
        print(f"      Completion: ${completion_price}")

def main():
    """Main function"""
    models = get_all_models()
    
    if not models:
        print("Failed to fetch models")
        return
    
    # Print by category
    print_models_by_category(models)
    
    # Print cheapest
    print_cheapest_models(models, count=20)
    
    # Save to CSV
    save_models_to_csv(models)
    
    print("\n" + "="*100)
    print("✅ COMPLETE - Check available_models_google_vertex.csv for full list")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
