import os, json, re

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs_json', 'mmlu_women_implicit_strong', '14')
count = 0
for fname in sorted(os.listdir(LOG_DIR)):
    if not fname.endswith('.json'):
        continue
    path = os.path.join(LOG_DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    # safe navigation
    resp = data.get('response')
    if resp and isinstance(resp, dict):
        choices = resp.get('choices')
        if choices and len(choices) > 0:
            msg = choices[0].get('message')
            if msg:
                content = msg.get('content')
                reasoning = msg.get('reasoning')
                if reasoning is None and content:
                    # fill reasoning with the content
                    msg['reasoning'] = content
                    changed = True
                # extract final answer pattern
                if 'final_answer' not in data and content:
                    m = re.search(r'Final answer:\s*([A-D])', content, re.IGNORECASE)
                    if m:
                        data['final_answer'] = m.group(1).upper()
                        changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        count += 1

print(f'Updated {count} files in {LOG_DIR}')
