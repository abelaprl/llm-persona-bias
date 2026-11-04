import json
from pathlib import Path
p=Path(r"D:\TugasAkhir\notebooks\gender\ethics accuracy analysis\ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb")
nb=json.loads(p.read_text(encoding="utf-8"))
old="""Coverage untuk keempat
subset, akurasi Commonsense/Justice, serta stance Deontology/Virtue dibaca
bersama tabel bagian 5."""
new="Dua metrik utama untuk keempat subset dibaca bersama tabel bagian 5."
for cell in nb["cells"]:
    if "source" in cell:
        s="".join(cell["source"])
        s=s.replace(old,new)
        cell["source"]=s.splitlines(keepends=True)
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
