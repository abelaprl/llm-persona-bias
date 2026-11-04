import os,re,tempfile
p=r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona\qwen_qwen3-next-80b-a3b-thinking\man_explicit_1\gsm8k_00041.json'
data=open(p,'rb').read(); new=re.sub(rb'("temperature"\s*:\s*)0\.0+',rb'\g<1>0',data)
tmp=p+'.codex_tmp'
open(tmp,'wb').write(new)
try:
 os.replace(tmp,p); print('replace_ok')
except Exception as e:
 print('replace_error',repr(e));
 try: os.unlink(tmp)
 except: pass
