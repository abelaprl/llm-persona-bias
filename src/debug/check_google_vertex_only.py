#!/usr/bin/env python3
"""
Filter ONLY Google Vertex provider models from OpenRouter
"""

import requests
import json

API_KEY = "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"

try:
    print("\n" + "="*120)
    print("Fetching ALL models from OpenRouter...")
    print("="*120 + "\n")
    
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        exit(1)
    
    all_models = response.json()['data']
    print(f"Total models from OpenRouter: {len(all_models)}\n")
    
    # Filter ONLY Google Vertex models
    # Google Vertex models have google/ prefix AND are Gemini/Nano Banana models
    google_vertex_models = []
    
    for model in all_models:
        model_id = model.get('id', '')
        
        # ONLY models from google/ provider
        if 'google/' in model_id.lower():
            google_vertex_models.append(model)
    
    print(f"\n{'='*120}")
    print(f"GOOGLE VERTEX PROVIDER MODELS: {len(google_vertex_models)}")
    print(f"{'='*120}\n")
    
    for i, model in enumerate(google_vertex_models, 1):
        model_id = model['id']
        name = model.get('name', 'N/A')
        pricing = model.get('pricing', {})
        prompt_price = pricing.get('prompt', 'N/A')
        completion_price = pricing.get('completion', 'N/A')
        
        print(f"[{i:2d}] {model_id:<50}")
        print(f"     Name: {name}")
        print(f"     Price: ${prompt_price} (prompt) | ${completion_price} (completion)\n")
    
    # Save to file
    output_file = "google_vertex_models_list.json"
    with open(output_file, 'w') as f:
        json.dump(google_vertex_models, f, indent=2)
    
    print(f"\n✅ Saved {len(google_vertex_models)} Google Vertex models to: {output_file}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
