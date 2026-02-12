"""
Fix OLMo 3 32B Think model - prompt id 2 only (copy reasoning to content)
"""
import json
from pathlib import Path

# Path to OLMo warmup file
olmo_path = Path("result/allenai_olmo-3-32b-think/man_implicit_2/warmup/persona_warmup.json")

if olmo_path.exists():
    print(f"Fixing: {olmo_path}")
    
    with open(olmo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if content is empty and reasoning exists
    if not data['response'].get('content') and data['response']['raw_api_response']['choices'][0]['message'].get('reasoning'):
        reasoning = data['response']['raw_api_response']['choices'][0]['message']['reasoning']
        
        # Copy reasoning to content
        data['response']['content'] = reasoning
        data['response']['raw_api_response']['choices'][0]['message']['content'] = reasoning
        
        # Save back
        with open(olmo_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Fixed - content copied from reasoning ({len(reasoning)} chars)")
    else:
        print(f"[SKIP] Already has content or no reasoning found")
else:
    print(f"[NOT FOUND] {olmo_path}")

print("\nDone!")
