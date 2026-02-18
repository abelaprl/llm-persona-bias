import re
from analyze_model31 import extract_final_answer

p='truncated_wrong_model34.txt'
lines=[l for l in open(p,encoding='utf-8').read().splitlines() if l.strip()]

pat_gt=re.compile(r'GT:([0-9eE\+\-\.]+)')
pat_snip=re.compile(r'Snippet:\s*(.*)')

results=[]
for ln in lines[:50]:
    mgt=pat_gt.search(ln)
    msk=pat_snip.search(ln)
    gt=float(mgt.group(1)) if mgt else None
    snippet=msk.group(1) if msk else ln
    ex=extract_final_answer(snippet)
    results.append({'line':ln,'gt':gt,'ex':ex,'snippet':snippet})

# summary
ok=0
none_count=0
for r in results:
    if r['ex'] is None:
        none_count+=1
    elif r['gt'] is not None and abs(r['ex']-r['gt'])<1e-9:
        ok+=1

print(f"Tested {len(results)} entries: exact matches={ok}, no_extract={none_count}")

print('\nSample mismatches:')
count=0
for r in results:
    if r['ex'] is None or (r['gt'] is not None and abs((r['ex'] or 0)-r['gt'])>1e-6):
        print('FILE SNIPPET:', r['line'].split('|')[0].strip())
        print(' GT=', r['gt'], ' EX=', r['ex'])
        print(' SNIPPET:', r['snippet'][:300])
        print('---')
        count+=1
        if count>=10:
            break
