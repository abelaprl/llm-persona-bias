#!/usr/bin/env python3
"""
List all models available from Google Vertex with simple output
"""

import requests
import json

API_KEY = "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"

def get_models():
    """Get all models"""
    print("\n🔍 Fetching models...\n")
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"❌ Error: {response.json()}")
            return []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def main():
    models = get_models()
    
    if not models:
        print("Failed to fetch models")
        return
    
    print(f"✅ Total {len(models)} models available\n")
    print("="*120)
    print(f"{'Model ID':<60} {'Prompt $/1k':<15} {'Completion $/1k':<15} {'Context'}")
    print("="*120)
    
    # Filter dan sort by price
    priced = []
    for m in models:
        pricing = m.get('pricing', {})
        prompt = pricing.get('prompt')
        completion = pricing.get('completion')
        
        if prompt is not None and completion is not None:
            priced.append((m['id'], prompt, completion, m.get('context_length', 'N/A')))
    
    priced.sort(key=lambda x: float(x[1]))
    
    for model_id, prompt, completion, context in priced[:100]:  # Show top 100 cheapest
        print(f"{model_id:<60} ${float(prompt):<14.10f} ${float(completion):<14.10f} {context}")
    
    print("\n" + "="*120)
    print(f"\n📊 Showing cheapest 100 out of {len(models)} models")
    print(f"\n💡 RECOMMENDED CHEAP MODELS FOR EXPERIMENTS:")
    print("="*120)
    
    recommendations = [
        m for m in priced 
        if any(keyword in m[0].lower() for keyword in ['mistral', 'deepseek', 'llama', 'qwen'])
    ]
    
    for i, (model_id, prompt, completion, context) in enumerate(recommendations[:20], 1):
        print(f"[{i:2d}] {model_id:<55} | Prompt: ${float(prompt):.10f} | Complete: ${float(completion):.10f}")
    
    # Save to file
    with open("google_vertex_models_list.json", 'w') as f:
        json.dump({
            'total': len(models),
            'models': [
                {
                    'id': m['id'],
                    'name': m.get('name'),
                    'prompt_price': m.get('pricing', {}).get('prompt'),
                    'completion_price': m.get('pricing', {}).get('completion'),
                    'context_length': m.get('context_length')
                }
                for m in models
            ]
        }, f, indent=2)
    
    print(f"\n✅ Full list saved to: google_vertex_models_list.json\n")

if __name__ == '__main__':
    main()
