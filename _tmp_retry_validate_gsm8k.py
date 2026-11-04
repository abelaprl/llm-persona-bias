import os,re,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
ROOT=r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona'
MAX_RE=re.compile(rb'("max_tokens"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
TEMP_RE=re.compile(rb'("temperature"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
paths=[]
for d,_,fs in os.walk(ROOT):
  for n in fs:
    if n.lower().endswith('.json'): paths.append(os.path.join(d,n))
print('files=',len(paths),flush=True)
def one(path):
  last=None
  for attempt in range(4):
    try:
      with open(path,'rb') as f:data=f.read()
      mm=list(MAX_RE.finditer(data)); tm=list(TEMP_RE.finditer(data))
      if len(mm)!=1 or len(tm)!=1:return ('missing',path,len(mm),len(tm))
      om=mm[0].group(2); ot=tm[0].group(2)
      if om==b'4096' and ot==b'0':return ('ok',path)
      new=TEMP_RE.sub(rb'\g<1>0',MAX_RE.sub(rb'\g<1>4096',data))
      with open(path,'wb') as f:f.write(new)
      return ('fixed',path,om.decode(),ot.decode(),attempt)
    except Exception as e:
      last=repr(e); time.sleep(0.02*(attempt+1))
  return ('error',path,last)
counts=Counter(); examples={}; oldm=Counter(); oldt=Counter(); start=time.time()
with ThreadPoolExecutor(max_workers=32) as ex:
  for i,r in enumerate(ex.map(one,paths,chunksize=50),1):
    counts[r[0]]+=1
    if r[0]=='fixed':oldm[r[2]]+=1;oldt[r[3]]+=1
    elif r[0] in ('missing','error') and r[0] not in examples:examples[r[0]]=r
    if i%10000==0:print(f'progress={i}/{len(paths)} counts={dict(counts)} elapsed={time.time()-start:.1f}s',flush=True)
print('completed counts=',dict(counts),'elapsed=',round(time.time()-start,1),flush=True)
print('old_max=',dict(oldm),flush=True);print('old_temp=',dict(oldt),flush=True);print('examples=',examples,flush=True)
