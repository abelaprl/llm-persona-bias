import json
from pathlib import Path
root=Path(r"D:\TugasAkhir")
new="Dua metrik utama dihitung pada keempat subset: akurasi penuh (n_correct/total) dan proporsi wrong pada respons committed (n_answer_wrong/committed). Grafik dua metrik tersedia pada 00_ethics_<domain>_two_metrics_all_subsets_heatmap."
for domain, names in {
 'gender':['ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb'],
 'religion':['ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb'],
 'regional':['ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb'],
}.items():
 folder=root/'notebooks'/domain/'ethics accuracy analysis'
 for name in names:
  p=folder/name
  raw=p.read_text(encoding='utf-8')
  raw=raw.replace('Akurasi hanya per subset Commonsense/Justice, dibaca bersama commit_rate. Deontology/Virtue memakai distribusi wrong/acceptable.',new)
  raw=raw.replace('- Akurasi tetap per subset untuk Commonsense/Justice; Deontology/Virtue memakai stance.',new)
  p.write_text(raw,encoding='utf-8')
