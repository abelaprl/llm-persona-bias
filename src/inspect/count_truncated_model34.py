import os
import json

root = os.path.join('result', 'openai_gpt-oss-120b')
length_counts = {}
heur_trunc_counts = {}
total = 0

for persona in sorted(os.listdir(root)):
    p = os.path.join(root, persona)
    if not os.path.isdir(p):
        continue
    lc = 0
    hc = 0
    for f in os.listdir(p):
        if not f.endswith('.json'):
            continue
        total += 1
        path = os.path.join(p, f)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            raw = d.get('response', {}).get('raw_api_response', {})
            choices = raw.get('choices', []) if raw else []
            fr = ''
            if choices:
                fr = choices[0].get('finish_reason', '')
            if fr == 'length':
                lc += 1
            content = d.get('response', {}).get('content', '')
            if content is None:
                hc += 1
            else:
                s = str(content).strip()
                if s == '' or (s[-1] not in '.!?)]}\"\'0123456789'):
                    hc += 1
        except Exception:
            # skip unreadable files
            pass
    length_counts[persona] = lc
    heur_trunc_counts[persona] = hc

print('Total files:', total)
print('\nFinish_reason==length per persona:')
for p in sorted(length_counts):
    print(f"{p}: {length_counts[p]}")
print('\nHeuristic truncated (ends-with check) per persona:')
for p in sorted(heur_trunc_counts):
    print(f"{p}: {heur_trunc_counts[p]}")

with open('count_truncated34.json', 'w', encoding='utf-8') as of:
    json.dump({'total_files': total, 'length_per_persona': length_counts, 'heur_trunc_per_persona': heur_trunc_counts}, of, ensure_ascii=False, indent=2)

print('\nWrote count_truncated34.json')
