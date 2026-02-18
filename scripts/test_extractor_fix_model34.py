import json
import os
from extractor34 import extract_final_answer

with open('analyze34_mismatches.json', 'r', encoding='utf-8') as f:
    mismatches = json.load(f)

sample = mismatches[:50]

results = []
correct = 0
no_extract = 0
changed_to_correct = 0

for it in sample:
    persona = it['persona']
    fname = it['file']
    gt = it.get('gt')
    path = os.path.join('result', 'openai_gpt-oss-120b', persona, fname)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        content = d.get('response', {}).get('content', '')
    except Exception as e:
        content = it.get('detail', '')

    new_ex = extract_final_answer(content)
    ok = (new_ex == gt)
    results.append({'persona': persona, 'file': fname, 'gt': gt, 'old_ex': it.get('extracted'), 'new_ex': new_ex, 'fixed': ok})
    if ok:
        changed_to_correct += 1
    if new_ex is None:
        no_extract += 1

with open('test_extractor_fix_results.json', 'w', encoding='utf-8') as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print('Sample size:', len(sample))
print('Changed to correct:', changed_to_correct)
print('No extract now:', no_extract)
print('Wrote test_extractor_fix_results.json')
