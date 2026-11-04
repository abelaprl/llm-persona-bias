import os, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
ROOT = r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona'
MAX_RE = re.compile(rb'("max_tokens"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
TEMP_RE = re.compile(rb'("temperature"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
paths=[]
for d, _, fs in os.walk(ROOT):
    for name in fs:
        if name.lower().endswith('.json'):
            paths.append(os.path.join(d,name))
print(f'files={len(paths)}', flush=True)

def update(path):
    try:
        with open(path,'rb') as f: data=f.read()
        mm=list(MAX_RE.finditer(data)); tm=list(TEMP_RE.finditer(data))
        if len(mm)!=1 or len(tm)!=1:
            return ('missing' if len(mm)==0 or len(tm)==0 else 'multi', path, len(mm), len(tm))
        old_max=mm[0].group(2); old_temp=tm[0].group(2)
        if old_max == b'4096' and old_temp == b'0':
            return ('unchanged', path, old_max.decode(), old_temp.decode())
        new=MAX_RE.sub(rb'\g<1>4096', data)
        new=TEMP_RE.sub(rb'\g<1>0', new)
        if new != data:
            with open(path,'wb') as f: f.write(new)
            return ('changed', path, old_max.decode(), old_temp.decode())
        return ('unchanged', path, old_max.decode(), old_temp.decode())
    except Exception as e:
        return ('error', path, repr(e))

counts=Counter(); old_max=Counter(); old_temp=Counter(); examples={}; start=time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    for i,res in enumerate(ex.map(update, paths, chunksize=20), 1):
        status=res[0]; counts[status]+=1
        if status in ('changed','unchanged'):
            old_max[res[2]]+=1; old_temp[res[3]]+=1
        elif status not in examples: examples[status]=res
        if i%10000==0:
            print(f'progress={i}/{len(paths)} changed={counts["changed"]} unchanged={counts["unchanged"]} missing={counts["missing"]} multi={counts["multi"]} errors={counts["error"]} elapsed={time.time()-start:.1f}s', flush=True)
print(f'completed counts={dict(counts)} elapsed={time.time()-start:.1f}s', flush=True)
print('old_max=',dict(old_max), flush=True)
print('old_temp=',dict(old_temp), flush=True)
print('examples=',examples, flush=True)
