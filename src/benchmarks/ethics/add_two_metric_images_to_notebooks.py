import json
from pathlib import Path
root=Path(r"D:\TugasAkhir")
for domain,name in {
 'gender':'ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb',
 'religion':'ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb',
 'regional':'ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb',
}.items():
 p=root/'notebooks'/domain/'ethics accuracy analysis'/name
 nb=json.loads(p.read_text(encoding='utf-8'))
 marker='## Dua metrik utama pada seluruh subset'
 for cell in nb.get('cells',[]):
  if marker in ''.join(cell.get('source',[])):
   source=''.join(cell['source'])
   image=f'![Dua metrik ETHICS {domain}](figures/00_ethics_{domain}_two_metrics_all_subsets_heatmap.png)'
   if image not in source:
    source += '\n\n' + image + '\n'
    cell['source']=source.splitlines(keepends=True)
   break
 p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
