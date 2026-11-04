import os,re,json
ROOT=r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona\qwen_qwen3-next-80b-a3b-thinking'
M=re.compile(rb'("max_tokens"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))');T=re.compile(rb'("temperature"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
counts={}; bad=[]
for d,_,fs in os.walk(ROOT):
 for n in fs:
  if n.endswith('.json'):
   p=os.path.join(d,n)
   try:data=open(p,'rb').read()
   except Exception as e: counts['errread']=counts.get('errread',0)+1; bad.append((p,'read',repr(e))); continue
   mm=list(M.finditer(data));tt=list(T.finditer(data));
   if len(mm)!=1 or len(tt)!=1: counts['missing']=counts.get('missing',0)+1; bad.append((p,len(mm),len(tt)))
   elif mm[0].group(2)!=b'4096' or tt[0].group(2)!=b'0': counts['non_target']=counts.get('non_target',0)+1; bad.append((p,mm[0].group(2),tt[0].group(2)))
   else:counts['ok']=counts.get('ok',0)+1
print(counts);print('bad_count',len(bad));print('first_bad',bad[:20])
