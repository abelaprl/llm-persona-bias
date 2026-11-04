"""Apply the frozen nine cultural-value measurements to both task surfaces."""
from pathlib import Path
import sys,json,hashlib,argparse
from dataclasses import fields,asdict
import numpy as np
import pandas as pd

ROOT=Path('D:/TugasAkhir');sys.path.insert(0,str(ROOT))
from src.core.linguistic_markers.registry import MarkerSpec,PatternSpec
from src.core.linguistic_markers.extractor import find_marker_matches
from src.benchmarks.ethics.harmonize_identity_markers import estimate,holm,contrasts

VERSION='cultural-values-common-20261015-v1'
RUNS=ROOT/'result/analyses/linguistic_markers/runs'
SOURCES={'gsm8k':'regional_gsm8k_raw_rebuild_20261005_231108_470909',
         'ethics':'regional_ethics_raw_rebuild_20261005_231140_525498'}
REGISTRY=[m for m in json.loads((RUNS/SOURCES['ethics']/'registry.json').read_text(encoding='utf-8')) if m['marker_id'].endswith('_orientation')]
IDS=[m['marker_id'] for m in REGISTRY]
assert len(IDS)==9
def specs():
    result=[]
    for m in REGISTRY:
        d={k:v for k,v in m.items() if k in {x.name for x in fields(MarkerSpec)}}
        d['patterns']=tuple(PatternSpec(**{k:v for k,v in p.items() if k in {'pattern','note','flags'}}) for p in m['patterns'])
        d['benchmarks']=('gsm8k','ethics');result.append(MarkerSpec(**d))
    return result
MARKERS=specs()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measure_text(text,word_count,available):
    metrics={};hits=[]
    for m in MARKERS:
        found=find_marker_matches(text,m) if available else []
        n=len(found) if available else np.nan
        metrics[m.marker_id+'__count']=n
        metrics[m.marker_id+'__density']=n/word_count*100 if available and word_count>0 else np.nan
        metrics[m.marker_id+'__present']=int(n>0) if available else np.nan
        hits += [dict(marker_id=m.marker_id,**asdict(h)) for h in found]
    return metrics,hits

def measure(out,ethics_only=False):
    out.mkdir(parents=True,exist_ok=ethics_only)
    (out/'registry.json').write_text(json.dumps(REGISTRY,indent=2),encoding='utf-8')
    manifest={'version':VERSION,'sources':{},'code_sha256':sha(__file__),
              'unit':'density per 100 explanation words','outcomes':IDS,'seed':20261015,
              'family':'9 densities x 52 contrasts x scopes; independently per model/task',
              'surface':'frozen analysis_text; explanation_available and word_count from the same source',
              'bootstrap':2000,'previous_results_modified':False}
    for task,run in SOURCES.items():
        source=RUNS/run/'response_records.csv.gz'
        if ethics_only and task=='gsm8k':
            manifest['sources'][task]={'path':str(source),'sha256':sha(source),'rows':len(pd.read_csv(out/'gsm8k_records.csv.gz',usecols=['source_row']))}
            continue
        cols=['model_id','persona','subset','question_id','question_hash','pair_eligible','analysis_text','word_count','explanation_available','final_text','question']
        if task=='ethics':cols += [mid+'__'+unit for mid in IDS for unit in ['count','density']]
        frame=pd.read_csv(source,usecols=cols,keep_default_na=False,low_memory=False)
        records=[];audit=[]
        for i,row in enumerate(frame.to_dict('records')):
            available=row['explanation_available'] in [True,'True',1] and bool(row['analysis_text'].strip())
            wc=float(row['word_count']) if str(row['word_count']).strip() else np.nan
            values,hits=measure_text(row['analysis_text'],wc,available)
            records.append({k:row[k] for k in ['model_id','persona','subset','question_id','question_hash','pair_eligible']}|{'source_row':i,'word_count':wc,'explanation_available':available}|values)
            if hits and task=='gsm8k':audit.append({k:row[k] for k in ['model_id','persona','question_id','question','final_text','analysis_text']}|{'source_row':i,'hits':hits})
            if i%40000==0:print(task,i,flush=True)
        measured=pd.DataFrame(records)
        if task=='ethics':
            checks=[]
            for mid in IDS:
                for unit in ['count','density']:
                    col=mid+'__'+unit;before=pd.to_numeric(frame[col],errors='coerce').to_numpy(float);after=measured[col].to_numpy(float)
                    same=np.isclose(before,after,rtol=1e-10,atol=1e-10,equal_nan=True)
                    checks.append({'marker_id':mid,'unit':unit,'mismatches':int((~same).sum())})
            pd.DataFrame(checks).to_csv(out/'ethics_reproduction.csv',index=False)
            assert not any(c['mismatches'] for c in checks),checks
        measured.to_csv(out/(task+'_records.csv.gz'),index=False,compression='gzip')
        if task=='gsm8k':
            (out/'gsm8k_full_hit_responses.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
            print('GSM8K distinct hit responses:',len(audit),flush=True)
            counts=pd.DataFrame([{'model_id':a['model_id'],'marker_id':h['marker_id']} for a in audit for h in a['hits']])
            if not counts.empty:print(counts.groupby(['model_id','marker_id']).size().to_string(),flush=True)
        manifest['sources'][task]={'path':str(source),'sha256':sha(source),'rows':len(frame)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def statistics(out):
    stats=[];profiles=[];rng=np.random.default_rng(20261015)
    for task in SOURCES:
        f=pd.read_csv(out/(task+'_records.csv.gz'))
        scopes=sorted(f['subset'].unique())+(['ethics_all'] if task=='ethics' else [])
        pairs=contrasts(f.persona.unique());assert len(pairs)==52
        for model in sorted(f.model_id.unique()):
            for scope in scopes:
                g=f[f.model_id.eq(model)]
                if scope!='ethics_all':g=g[g['subset'].eq(scope)]
                for mid in IDS:
                    for persona,part in g.groupby('persona'):
                        profiles.append({'task':task,'model_id':model,'subset':scope,'persona':persona,'marker_id':mid,
                            'n_available':int(part[mid+'__density'].count()),'mean_density':part[mid+'__density'].mean(),
                            'occurrences':part[mid+'__count'].sum(),'responses_present':int(part[mid+'__count'].gt(0).sum()),
                            'prevalence':part[mid+'__present'].mean()})
                eligible=g[g.pair_eligible.eq(True)]
                keys=['subset','question_id','question_hash'];cols=keys+[m+'__density' for m in IDS]
                for left,right,comparison in pairs:
                    paired=eligible[eligible.persona.eq(left)][cols].merge(eligible[eligible.persona.eq(right)][cols],on=keys,suffixes=('_left','_right'),validate='one_to_one')
                    for mid in IDS:
                        result=estimate(paired[mid+'__density_left'].to_numpy(float),paired[mid+'__density_right'].to_numpy(float),False,rng)
                        stats.append(dict(task=task,model_id=model,subset=scope,left=left,right=right,comparison=comparison,marker_id=mid,**result))
            print('Statistics',task,model,flush=True)
    s=pd.DataFrame(stats)
    for (task,model),idx in s.groupby(['task','model_id']).groups.items():
        size=9*52*(5 if task=='ethics' else 1);assert len(idx)==size
        s.loc[idx,'p_holm']=holm(s.loc[idx,'p_raw'].to_numpy(float));s.loc[idx,'holm_family_size']=size
    s.to_csv(out/'paired_statistics.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'profiles.csv',index=False)
    print(s.groupby('task').agg(tests=('p_raw','size'),significant=('p_holm',lambda x:int(x.lt(.05).sum()))).to_string(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['measure','measure_ethics','stats','all'],default='all');args=parser.parse_args()
    if args.phase in ['measure','all']:measure(args.output)
    if args.phase=='measure_ethics':measure(args.output,ethics_only=True)
    if args.phase in ['stats','all']:statistics(args.output)
