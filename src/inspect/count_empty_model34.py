import os
import json
from collections import defaultdict

root = os.path.join('result', 'openai_gpt-oss-120b')
per_persona = defaultdict(int)
per_persona_total = defaultdict(int)
total = 0
empty_total = 0
errors = 0

for persona in sorted(os.listdir(root)):
    ppath = os.path.join(root, persona)
    if not os.path.isdir(ppath):
        continue
    for fname in os.listdir(ppath):
        if not fname.endswith('.json'):
            continue
        total += 1
        per_persona_total[persona] += 1
        fpath = os.path.join(ppath, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = data.get('response', {}).get('content', '')
            if not content or str(content).strip() == '':
                per_persona[persona] += 1
                empty_total += 1
        except Exception:
            errors += 1

print('Per-persona empty counts:')
for p in sorted(per_persona_total.keys()):
    print(f"{p}: {per_persona.get(p,0)} / {per_persona_total[p]}")

print('\nTOTAL files:', total)
print('TOTAL EMPTY:', empty_total)
print('ERRORS:', errors)

# write JSON summary
out = {
    'total_files': total,
    'total_empty': empty_total,
    'errors': errors,
    'per_persona_total': dict(per_persona_total),
    'per_persona_empty': dict(per_persona),
}
with open('count_empty34.json', 'w', encoding='utf-8') as of:
    json.dump(out, of, ensure_ascii=False, indent=2)

print('\nWrote count_empty34.json')
import json,os
MODEL_DIR='result/openai_gpt-oss-120b'
personas=['man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3','woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3']

total=0
empty=0
per={}

for p in personas:
    path=os.path.join(MODEL_DIR,p)
    if not os.path.isdir(path):
        per[p]=0
        continue
    cnt=0
    files=[f for f in os.listdir(path) if f.endswith('.json')]
    for fn in files:
        total+=1
        try:
            with open(os.path.join(path,fn),'r',encoding='utf-8') as f:
                d=json.load(f)
            c=d.get('response',{}).get('content','')
            if not c or c.strip()=='':
                empty+=1
                cnt+=1
        except Exception as e:
            pass
    per[p]=cnt

print('Per persona empty:')
for k,v in per.items(): print(f"{k}: {v}")
print('TOTAL empty:',empty,'TOTAL files:',total)
