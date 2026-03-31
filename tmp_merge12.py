import csv
from pathlib import Path

reports = [
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/allenai_olmo-3-32b-think"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/deepseek_deepseek-v3.2"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/google_gemini-2.5-flash"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/google_gemma-2-9b-it"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/google_gemma-3-27b-it"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/meta-llama_llama-3.1-8b-instruct"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/meta-llama_llama-3.3-70b-instruct"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/mistralai_ministral-8b-2512"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/openai_gpt-oss-20b"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/openai_gpt-oss-120b"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/qwen_qwen3-235b-a22b-2507"),
    Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/qwen_qwen3-next-80b-a3b-thinking"),
]
required = {'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3','woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3','neutral'}
exclude = ['results_man_','results_woman_','results_neutral','accuracy','backup','compare','changed_cases','all_models_all_personas','from_rebuild_results_csv']

def inspect(path):
    try:
        with path.open(encoding='utf-8') as f:
            r = csv.DictReader(f)
            rows = list(r)
            if not r.fieldnames:
                return None
    except Exception:
        return None
    personas = {x.get('persona_folder','').strip() for x in rows if x.get('persona_folder')}
    cov = len(required & personas)
    n = path.name.lower()
    pref = 2 if n.startswith('result_') else (1 if n.startswith('results_') else 0)
    return {'path': path, 'rows': rows, 'fields': list(r.fieldnames), 'cov': cov, 'missing': sorted(required-personas), 'nrows': len(rows), 'pref': pref}

selected=[]; issues=[]
for d in reports:
    cands=[]
    if d.exists():
        for p in d.iterdir():
            if not p.is_file() or not p.name.lower().endswith('.csv'):
                continue
            n=p.name.lower()
            if any(t in n for t in exclude):
                continue
            if n.startswith('result_') or n.startswith('results_') or n=='results.csv':
                info=inspect(p)
                if info: cands.append(info)
    if not cands:
        issues.append((d.name,'no_candidate_csv','',0,sorted(required))); continue
    cands.sort(key=lambda x:(x['cov'],x['nrows'],x['pref']), reverse=True)
    best=cands[0]; selected.append((d.name,best))
    if best['cov']<9:
        issues.append((d.name,'incomplete_personas',best['path'].name,best['cov'],best['missing']))

out_csv=Path(r"e:/Tugas Akhir/TugasAkhir/result/reports/results_12_models_all_personas.csv")
union=['model_name']
for _,b in selected:
    for c in b['fields']:
        if c not in union: union.append(c)

count=0
with out_csv.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=union); w.writeheader()
    for model,b in selected:
        for row in b['rows']:
            o={k:'' for k in union}; o['model_name']=model
            for k,v in row.items():
                if k in o: o[k]=v
            w.writerow(o); count+=1

print('OUTPUT='+str(out_csv))
print('MODELS_SELECTED='+str(len(selected)))
print('ROWS_WRITTEN='+str(count))
for model,b in selected:
    print(f"SELECTED::{model}::{b['path'].name}::coverage={b['cov']}/9::rows={b['nrows']}")
print('ISSUES='+str(len(issues)))
for m,t,fn,cov,miss in issues:
    print(f"ISSUE::{m}::{t}::{fn}::coverage={cov}/9::missing={';'.join(miss)}")
