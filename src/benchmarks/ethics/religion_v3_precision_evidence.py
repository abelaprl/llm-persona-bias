"""Bind sampled raw spans to readable final-only precision evidence."""
from bisect import bisect_left,bisect_right
import hashlib
import json
import random
from collections import Counter,defaultdict
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import SOURCE_COLUMNS,KEYS
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
from src.benchmarks.ethics.religion_lm_v3_io import write_json


def sample_global_occurrences(path,all_rules,seed=18222008):
    rng=random.Random(seed);counts=Counter();reservoir=defaultdict(list)
    allowed=set(all_rules)
    for chunk in pd.read_csv(path,chunksize=50000,keep_default_na=False):
        columns=chunk.columns.tolist();rule_index=columns.index('rule_id')
        for values in chunk.itertuples(index=False,name=None):
            rule=values[rule_index]
            if rule not in allowed:raise ValueError('Unregistered occurrence rule')
            counts[rule]+=1;n=counts[rule]
            index=n-1 if n<=100 else rng.randrange(n)
            if index<100:
                row=dict(zip(columns,values))
                if n<=100:reservoir[rule].append(row)
                else:reservoir[rule][index]=row
    summaries=[];selected=[]
    for rule in all_rules:
        sample=reservoir[rule];selected.extend(sample)
        summaries.append(dict(rule_id=rule,total_rule_hits=counts[rule],sampled_occurrences=len(sample),
            sampled_unique_responses=len({r['response_key'] for r in sample}),human_reviewed=0,false_positives=None,
            status='pending_human_review' if counts[rule] else 'no_hit_precision_not_estimable'))
    return pd.DataFrame(selected),pd.DataFrame(summaries)


def precision_evidence(source,selected):
    selected=selected.copy()
    keys=set(selected.response_key)
    sampled_spans={key:group[['raw_start','raw_end','raw_matched_text']].to_dict('records') for key,group in selected.groupby('response_key')}
    evidence={};surfaces={}
    for chunk in pd.read_csv(source,usecols=SOURCE_COLUMNS,dtype=str,keep_default_na=False,chunksize=5000):
        chunk=chunk.rename(columns={'model':'model_id','persona_folder':'persona','response_content_full':'final_output'})
        for values,text in zip(chunk[KEYS].itertuples(index=False,name=None),chunk.final_output):
            key=json.dumps(list(values),separators=(',',':'))
            if key not in keys:continue
            for span in sampled_spans[key]:
                if text[span['raw_start']:span['raw_end']]!=span['raw_matched_text']:
                    raise ValueError('Precision raw span disagrees with source')
            surface=prepare_final_surface(text)
            surfaces[key]=([a for a,b in surface.maps['visible']],[b for a,b in surface.maps['visible']])
            evidence[key]=dict(response_key=key,final_output=surface.visible_text,
                raw_final_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                offset_note='Precision sample includes visible_start/end and raw_start/end; displayed text excludes hidden blocks.')
    if set(evidence)!=keys:raise ValueError('Incomplete precision evidence')
    starts,ends=[],[]
    for r in selected.itertuples(index=False):
        a,b=surfaces[r.response_key]
        start,end=bisect_right(b,r.raw_start),bisect_left(a,r.raw_end)
        if start>=end:raise ValueError('Precision span has no visible final text')
        if r.raw_final_sha256!=evidence[r.response_key]['raw_final_sha256']:raise ValueError('Precision source hash mismatch')
        starts.append(start);ends.append(end)
    selected['visible_start']=starts;selected['visible_end']=ends
    return selected,[evidence[k] for k in sorted(keys)]


def write_precision_templates(output,selected,evidence):
    key_ids={r['response_key']:f'P{i:05d}' for i,r in enumerate(evidence,1)}
    blind=[dict(audit_id=key_ids[r['response_key']],final_output=r['final_output'],raw_final_sha256=r['raw_final_sha256']) for r in evidence]
    write_json(output/'audit/precision_full_blind.json',blind)
    key=selected.copy()
    key['sample_id']=[f'S{i:06d}' for i in range(1,len(key)+1)]
    key['audit_id']=key.response_key.map(key_ids)
    key.to_csv(output/'audit/precision_design_key.csv',index=False)
    template=key[['sample_id','audit_id','marker_id','visible_start','visible_end']].copy()
    for column in ('correct_match','reviewer_id','full_response_read','error_type','notes'):template[column]=''
    for reviewer in ('reviewer_a','reviewer_b','adjudication'):
        directory=output/'audit'/reviewer;directory.mkdir()
        template.to_csv(directory/'precision_annotations.csv',index=False)
    with (output/'audit/precision_README.md').open('x',encoding='utf-8') as handle:
        handle.write('# Audit precision kandidat\n\nBaca `precision_full_blind.json` bersama definisi marker di registry. `visible_start/end` memakai offset karakter Unicode pada teks final yang ditampilkan; span akhir eksklusif. Baca seluruh respons sebelum menilai span. Isi correct_match dengan 0/1, identitas reviewer sebenarnya, full_response_read, tipe kesalahan dan alasan. Blank bukan nol.\n\nReviewer A dan B bekerja independen sebelum adjudikasi. `precision_design_key.csv` menyimpan provenance model/persona/rule/raw span untuk evaluasi; jangan memakai identitas model atau ground truth jawaban untuk memilih label marker. Paket precision tidak menggantikan recall tanpa anchor di paket held-out terpisah. Seluruh berkas saat dibuat adalah template kosong, bukan gold.\n')
