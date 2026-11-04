import os,re,time
ROOT=r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona\qwen_qwen3-next-80b-a3b-thinking'
T=re.compile(rb'("temperature"\s*:\s*)0\.0+\b')
changed=ok=bad=0; examples=[]
for d,_,fs in os.walk(ROOT):
 for n in fs:
  if not n.endswith('.json'):continue
  p=os.path.join(d,n)
  try:data=open(p,'rb').read()
  except Exception as e:bad+=1;examples.append((p,'read',repr(e)));continue
  if not T.search(data):
   ok+=1;continue
  new=T.sub(rb'\g<1>0',data)
  try:
   with open(p,'wb') as f:f.write(new)
   changed+=1
  except Exception as e:bad+=1;examples.append((p,'write',repr(e)))
print({'changed':changed,'already_ok':ok,'errors':bad});print('examples',examples[:10])
