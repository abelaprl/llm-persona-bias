import json
import os

MODEL_DIR = 'result/allenai_olmo-3-32b-think'
PERSONAS = [
    'man_explicit_1', 'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'woman_explicit_1', 'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
]

empty_count = 0
total_count = 0

print("Empty files per persona:")
print("-" * 40)

for persona in PERSONAS:
    persona_path = os.path.join(MODEL_DIR, persona)
    if not os.path.exists(persona_path):
        continue
    
    persona_empty = 0
    files = [f for f in os.listdir(persona_path) if f.endswith('.json')]
    
    for fname in files:
        fpath = os.path.join(persona_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = data.get('response', {}).get('content', '')
            total_count += 1
            if not content or content.strip() == '':
                empty_count += 1
                persona_empty += 1
        except Exception as e:
            pass
    
    print(f"{persona}: {persona_empty} empty")

print("-" * 40)
print(f"TOTAL: {empty_count} empty files out of {total_count}")
print(f"Estimated time: {empty_count * 2 / 60:.1f} minutes (2 sec/request)")
print(f"Estimated cost: ~${empty_count * 0.001:.2f} (at ~$0.001/request)")
