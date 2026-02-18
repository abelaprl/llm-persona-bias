import importlib.util
import pathlib
import json
import os

spec = importlib.util.spec_from_file_location('extractor34', str(pathlib.Path(__file__).parent.parent / 'extractor34.py'))
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

with open(str(pathlib.Path(__file__).parent.parent / 'analyze34_mismatches.json'), 'r', encoding='utf-8') as f:
    mismatches = json.load(f)

sample = mismatches[:50]

results = []
changed_to_correct = 0
no_extract = 0

for it in sample:
    persona = it['persona']
    fname = it['file']
    gt = it.get('gt')
    path = os.path.join('result', 'openai_gpt-oss-120b', persona, fname)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        content = d.get('response', {}).get('content', '')
    except Exception:
        content = it.get('detail', '')

    new_ex = extractor.extract_final_answer(content)
    ok = (new_ex == gt)
    results.append({'persona': persona, 'file': fname, 'gt': gt, 'old_ex': it.get('extracted'), 'new_ex': new_ex, 'fixed': ok})
    if ok:
        changed_to_correct += 1
    if new_ex is None:
        no_extract += 1

outpath = pathlib.Path(__file__).parent / 'test_extractor_fix_results.json'
with open(outpath, 'w', encoding='utf-8') as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print('Sample size:', len(sample))
print('Changed to correct:', changed_to_correct)
print('No extract now:', no_extract)
print('Wrote', outpath)
