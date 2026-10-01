"""Scan observed lexical themes before adding rules; never read answer labels."""
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import load_final_rows,write_json,create_run_directory

THEMES = {
 'theme_harm':(r'\b(?:harm|harmful|suffering)\b','O1-01/O1-07/O2-60'),
 'theme_compassion':(r'\b(?:compassion|empathy|kindness)\b','O1-04/O1-11/O2-60'),
 'theme_fairness':(r'\b(?:fairness|equality|inequity)\b','O2-17/O2-33'),
 'theme_consent_autonomy':(r'\b(?:consent|autonomy)\b','O1-10/O1-19/O2-39'),
 'theme_boundaries':(r'\b(?:boundary|boundaries)\b','R051/O1-08/O1-43/O2-59'),
 'theme_responsibility':(r'\b(?:responsibility|responsibilities|accountability)\b','O1-13/O2-56/O2-59'),
 'theme_intention':(r'\b(?:intent|intention|intentions)\b','O1-02/O1-33/O2-39'),
 'theme_honesty':(r'\b(?:honesty|dishonesty|deception)\b','O1-23/O2-39'),
}

def main():
    base = ROOT/'result/ETHICS/reports'
    prep = base/'religion_lm_v3_preparation_20260915_r1'
    out = create_run_directory(base/'religion_lm_v3_theme_discovery_20260915_r1')
    rows = load_final_rows(base/'Religion_Persona/_combined/all_models_results_full.csv')
    split = pd.read_csv(base/'religion_lm_v3_partition_revision_20260915_r1/question_split.csv')
    rows = rows.merge(split,on=['subset','question_id'],validate='many_to_one')
    read = []
    for name in ('targeted_full_responses.json','open_discovery_round_1.json','open_discovery_round_2.json'):
        read.extend(json.loads((prep/name).read_text(encoding='utf-8')))
    read.extend(json.loads((base/'religion_lm_v3_partition_revision_20260915_r1/rare_full_responses.json').read_text(encoding='utf-8')))
    prior = {r['response_key'] for r in read}
    probes, inventory = {}, []
    for name,(pattern,source) in THEMES.items():
        counts = rows.final_output.str.count(re.compile(pattern,re.I))
        probes[name] = counts.gt(0)
        scan = rows[['model_id','persona','subset']].copy()
        scan['response_hits'] = counts.gt(0).astype(int)
        scan['occurrence_hits'] = counts
        scan = scan.groupby(['model_id','persona','subset']).sum().reset_index()
        scan['form_id'],scan['pattern'],scan['source_sample_id'] = name,pattern,source
        inventory.append(scan)
    pd.concat(inventory).to_csv(out/'form_inventory.csv',index=False)
    selected, quota = [], []
    for model,group in rows.groupby('model_id',sort=True):
        existing = group.response_key.isin(prior)
        needed = {name:max(0,min(10,int(probes[name].loc[group.index].sum()))-int((probes[name].loc[group.index]&existing).sum())) for name in THEMES}
        pool = group.loc[group.partition.eq('development') & ~existing].sample(frac=1,random_state=18222008)
        while any(needed.values()):
            score = sum(probes[name].loc[pool.index].astype(int) for name,n in needed.items() if n>0)
            if score.max()==0:
                raise ValueError('Development pool cannot meet readback quota')
            index = score.idxmax()
            record = pool.loc[index].to_dict()
            record['form_ids'] = [name for name in THEMES if probes[name].loc[index]]
            selected.append(record)
            for name in record['form_ids']:
                needed[name] = max(0,needed[name]-1)
            pool = pool.drop(index)
        for name in THEMES:
            keys = prior|{r['response_key'] for r in selected}
            quota.append(dict(model_id=model,form_id=name,total_response_hits=int(probes[name].loc[group.index].sum()),
                readback_planned=int((probes[name].loc[group.index]&group.response_key.isin(keys)).sum()),human_reviewed=0))
    for i,record in enumerate(selected):
        record['readback_id'] = f'T{i+1:03d}'
    write_json(out/'full_responses.json',selected)
    pd.DataFrame(quota).to_csv(out/'readback_quotas.csv',index=False)
    write_json(out/'manifest.json',dict(status='discovery_pending_readback',new_responses=len(selected),
        lexical_only=True,no_stance_or_religion_inference=True,selection='greedy multi-form quota; not precision or recall sample'))
    print(f'Theme queue: {len(selected)} complete responses',flush=True)

if __name__=='__main__':
    main()
