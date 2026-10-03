"""Summarize measured changes and discovery support without claiming human gold."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import KEYS,create_run_directory,sha256,write_json,build_source_manifest
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,BASE_PATTERN_COUNTS


def summarize(report,output):
    output=create_run_directory(output)
    markers=[m.marker_id for m in get_registry()]
    base=ROOT/'result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
    old=pd.read_csv(base,usecols=lambda c:c in KEYS+['text_available','word_count'] or c.endswith('__count')).set_index(KEYS)
    totals=Counter(); replay=[]
    for chunk in pd.read_csv(report/'comparison_with_baseline/factorial_ablation.csv.gz',chunksize=5000):
        chunk=chunk.set_index(KEYS)
        previous=old.loc[chunk.index]
        for model,group in chunk.groupby(level='model_id'):
            for marker in markers:
                is_old=marker in BASE_PATTERN_COUNTS
                def col(name):return group[name].fillna(0).to_numpy(float)
                zero=np.zeros(len(group))
                c00=col('ablation_old_surface_old_registry__'+marker) if is_old else zero
                c10=col('ablation_new_surface_old_registry__'+marker) if is_old else zero
                c01=col('ablation_old_surface_new_registry__'+marker)
                c11=col(marker+'__count')
                for metric,value in {'old_surface_old_registry':c00.sum(),'new_surface_old_registry':c10.sum(),
                    'old_surface_new_registry':c01.sum(),'new_surface_new_registry':c11.sum(),
                    'surface_scope_delta_at_old_registry':(c10-c00).sum(),'registry_delta_at_new_surface':(c11-c10).sum(),
                    'interaction':(c11-c10-c01+c00).sum(),'changed_responses':np.count_nonzero(c11!=c00)}.items():
                    totals[(model,marker,metric)]+=value
                if is_old:
                    expected=previous.loc[group.index,marker+'__count'].fillna(0).to_numpy(float)
                    totals[(model,marker,'baseline_replay_mismatch_responses')]+=np.count_nonzero(c00!=expected)
    pd.DataFrame([dict(model_id=m,marker_id=k,metric=v,value=n) for (m,k,v),n in totals.items()]).to_csv(output/'factorial_ablation_summary.csv',index=False)
    forms=Counter(); rule_responses=defaultdict(set); scope=Counter(); new_hits=Counter()
    for chunk in pd.read_csv(report/'occurrences/hits.csv.gz',chunksize=50000,keep_default_na=False):
        for r in chunk.itertuples(index=False):
            form=re.sub(r'\s+',' ',r.matched_text).strip().casefold()
            forms[(r.marker_id,form)]+=1
            rule_responses[(r.model_id,r.rule_id)].add(r.response_key)
            scope[(r.model_id,r.speaker_role,r.discourse_scope)]+=1
            if ':candidate:' in r.rule_id:new_hits[(r.model_id,r.rule_id)]+=1
    pd.DataFrame([dict(marker_id=m,observed_form=f,occurrences=n) for (m,f),n in forms.items()]).to_csv(output/'observed_forms.csv.gz',index=False)
    pd.DataFrame([dict(model_id=m,speaker_role=s,discourse_scope=d,occurrences=n) for (m,s,d),n in scope.items()]).to_csv(output/'scope_counts.csv',index=False)
    readback=pd.read_csv(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_development_review_20260915_r1/readback_by_model_rule.csv').set_index(['model_id','rule_id'])
    support=[]
    for (model,rule),keys in sorted(rule_responses.items()):
        read=int(readback.loc[(model,rule),'ai_full_responses_read']) if (model,rule) in readback.index else 0
        support.append(dict(model_id=model,rule_id=rule,hit_responses=len(keys),ai_full_responses_read=read,
            required_development_readback=min(10,len(keys)),development_quota_met=read>=min(10,len(keys)),
            origin='candidate' if ':candidate:' in rule else 'legacy',human_reviewed=0,false_positives=None,
            status='candidate_not_human_validated'))
    support=pd.DataFrame(support);support.to_csv(output/'rule_readback_support.csv',index=False)
    matrix=pd.read_csv(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_development_review_20260915_r1/completeness_matrix.csv')
    count=Counter(m for m,f in forms)
    matrix['form_count_observed']=matrix.marker_id.map(count).fillna(0).astype(int)
    matrix['form_count_definition']='distinct whitespace-normalized casefolded accepted matched spans; not synonym or semantic coverage'
    matrix.to_csv(output/'completeness_matrix.csv',index=False)
    summary=pd.read_csv(report/'statistics/overall_descriptive.csv')
    inventory=pd.read_csv(report/'inventory/coverage.csv')
    inventory.groupby(['model_id','surface_status']).total.sum().unstack(fill_value=0).to_csv(output/'surface_counts_by_model.csv')
    outcome=summary.groupby(['model_id','marker_id']).agg(raw_count=('raw_count','sum'),word_total=('word_total','sum'),
        linguistic_responses=('linguistic_available','sum'),final_responses=('final_available','sum')).reset_index()
    outcome.to_csv(output/'marker_totals_by_model.csv',index=False)
    comparisons=pd.read_csv(report/'statistics/paired_contrasts.csv.gz')
    sensitivity=pd.read_csv(report/'statistics/sensitivity_directions.csv.gz')
    reversals=sensitivity.groupby(['population','metric']).agg(total=('direction_evaluable','size'),
        evaluable=('direction_evaluable','sum'),direction_reversed=('direction_reversed','sum')).reset_index()
    reversals.to_csv(output/'sensitivity_direction_summary.csv',index=False)
    from src.benchmarks.ethics.religion_v3_gap_provenance import enrich_gap_provenance
    unlocated=enrich_gap_provenance(output)
    result=dict(status='candidate_descriptive_audit_not_precision_recall',new_rule_hits=[dict(model_id=m,rule_id=r,occurrences=n) for (m,r),n in new_hits.items()],
        baseline_replay_mismatch_marker_response_cells=int(sum(v for (m,k,t),v in totals.items() if t=='baseline_replay_mismatch_responses')),
        new_rule_model_readback_quota_failures=support.loc[support.origin.eq('candidate')&~support.development_quota_met].to_dict('records'),
        observed_forms=len(forms),occurrences=sum(forms.values()),inference_scope_counts=comparisons.inference_scope_status.value_counts().to_dict(),
        report_manifest_sha256=sha256(report/'manifest.json'),script_sha256=sha256(__file__),human_reviews_completed=0,
        gap_evidence_anchors_not_located=unlocated)
    result['audit_sources']=build_source_manifest([Path(__file__),ROOT/'src/benchmarks/ethics/religion_v3_gap_provenance.py'])
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='new_rule_hits'},default=str),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();summarize(args.report,args.output)
