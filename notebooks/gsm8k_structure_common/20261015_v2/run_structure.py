"""Offline common GSM8K structure cues. No generation or answer scoring."""
from pathlib import Path
import argparse, hashlib, json, sys, gzip
import numpy as np
import pandas as pd
from structure_cues import IDS, LABELS, detect, flatten
ROOT=Path('D:/TugasAkhir')
sys.path.insert(0,str(ROOT))
from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
from src.benchmarks.ethics.harmonize_identity_markers import contrasts, estimate, holm

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=False)
 here=Path(__file__).parent
 sources=json.loads((here/'sources.json').read_text())
 manifest=dict(version='gsm8k-structure-common-20261015-v2',sources=sources,seed=20261015,
  denominator='visible nonempty final response; unknown surface excluded; answer-only included',
  measurement='binary presence of four literal cues; not reconstruction of original layout',
  family='four cues times all contrasts, separately per domain/model',
  bootstrap=2000,code_sha256=sha(__file__),rules_sha256=sha(here/'structure_cues.py'))
 stats=[];profiles=[];rng=np.random.default_rng(20261015)
 for domain,source in sources.items():
  assert sha(source['path'])==source['sha256']
  col='raw_final_text' if domain=='gender' else 'final_text'
  cols=['model_id','persona','question_id','question_hash','pair_eligible','raw_source_file',col]
  frame=pd.read_csv(source['path'],usecols=cols,keep_default_na=False,low_memory=False)
  records=[]
  with gzip.open(out/(domain+'_matches.jsonl.gz'),'wt',encoding='utf-8') as f:
   for i,row in enumerate(frame.to_dict('records')):
    raw=row.pop(col);surface=build_surfaces(raw);visible=surface.visible_text
    available=bool(visible.strip()) and surface.status!='unknown_surface'
    flat=flatten(visible);hits=detect(visible) if available else []
    assert hits==(detect(flat) if available else [])
    row.update(source_row=i,visible_available=available,surface_status=surface.status,
     final_sha256=hashlib.sha256(raw.encode()).hexdigest(),flat_sha256=hashlib.sha256(flat.encode()).hexdigest())
    row['pair_eligible']=str(row['pair_eligible']).lower()=='true'
    for m in IDS:
     count=sum(h['marker_id']==m for h in hits)
     row[m+'__present']=int(count>0) if available else np.nan
     row[m+'__count']=count if available else np.nan
    if hits:f.write(json.dumps(dict(source_row=i,model_id=row['model_id'],persona=row['persona'],question_id=row['question_id'],hits=hits))+'\n')
    records.append(row)
  frame=pd.DataFrame(records)
  assert len(frame)==source['rows']
  assert not frame[frame.pair_eligible].duplicated(['model_id','persona','question_id','question_hash']).any()
  frame.to_csv(out/(domain+'_records.csv.gz'),index=False)
  for model,g in frame.groupby('model_id'):
   for persona,part in g.groupby('persona'):
    for m in IDS:
     v=part[m+'__present'].dropna()
     profiles.append(dict(domain=domain,model_id=model,persona=persona,marker_id=m,n_available=len(v),n_present=v.sum(),percentage=v.mean()*100))
   eligible=g[g.pair_eligible]
   for left,right,comparison in contrasts(g.persona.unique()):
    keys=['question_id','question_hash'];c=keys+[m+'__present' for m in IDS]
    pair=eligible[eligible.persona.eq(left)][c].merge(eligible[eligible.persona.eq(right)][c],on=keys,suffixes=('_left','_right'),validate='one_to_one')
    for m in IDS:
     result=estimate(pair[m+'__present_left'].to_numpy(float),pair[m+'__present_right'].to_numpy(float),True,rng)
     stats.append(dict(domain=domain,model_id=model,left=left,right=right,comparison=comparison,marker_id=m,**result))
  assert sha(source['path'])==source['sha256']
  print(domain,len(frame),'rows complete; whitespace invariance and source hashes pass',flush=True)
 stats=pd.DataFrame(stats)
 for (domain,model),indices in stats.groupby(['domain','model_id']).groups.items():
  size=72 if domain=='gender' else 208
  assert len(indices)==size
  stats.loc[indices,'p_holm']=holm(stats.loc[indices,'p_raw'].to_numpy(float))
  stats.loc[indices,'holm_family_size']=size
 assert len(stats)==2928
 stats.to_csv(out/'paired_statistics.csv',index=False)
 pd.DataFrame(profiles).to_csv(out/'profiles.csv',index=False)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('Completed',len(stats),'paired comparisons',flush=True)

if __name__=='__main__':main()
