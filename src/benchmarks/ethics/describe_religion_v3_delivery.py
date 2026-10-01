"""Measured report tables; no scoring or human annotation is performed."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory,write_json,sha256
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,BASE_PATTERN_COUNTS


def describe(report,output):
    output=create_run_directory(output)
    markers=[m.marker_id for m in get_registry()]
    columns=['model_id','persona','surface_status','explanation_available','final_output_available','answer_status','word_count']+[m+'__count' for m in markers]
    frame=pd.read_csv(report/'response_level/markers.csv.gz',usecols=columns,dtype={m+'__count':float for m in markers},low_memory=False)
    frame['abstain_with_text']=frame.explanation_available&frame.answer_status.eq('abstain')
    frame['marker_occurrences']=frame[[m+'__count' for m in markers]].sum(axis=1,min_count=1)
    new=[m for m in markers if m not in BASE_PATTERN_COUNTS]
    frame['new_marker_occurrences']=frame[[m+'__count' for m in new]].sum(axis=1,min_count=1)
    model=frame.groupby('model_id').agg(total=('persona','size'),linguistic_text=('explanation_available','sum'),
        available_final=('final_output_available','sum'),abstain_with_text=('abstain_with_text','sum'),
        word_total=('word_count','sum'),marker_occurrences=('marker_occurrences','sum'),new_marker_occurrences=('new_marker_occurrences','sum'))
    surfaces=pd.crosstab(frame.model_id,frame.surface_status)
    for name in ('label_only','empty_final','technical_only'):model[name]=surfaces[name] if name in surfaces else 0
    model['text_rate']=model.linguistic_text/model.total
    model.to_csv(output/'denominators_by_model.csv')
    summary=pd.read_csv(report/'statistics/overall_descriptive.csv')
    totals=summary.groupby('marker_id').raw_count.sum().rename('raw_count').reset_index()
    totals['added_in_v3']=totals.marker_id.isin(new)
    totals.to_csv(output/'marker_totals.csv',index=False)
    contrasts=pd.read_csv(report/'statistics/paired_contrasts.csv.gz')
    selected=contrasts.loc[contrasts.population.eq('text_all')&contrasts.metric.eq('prevalence_text')&
        contrasts.contrast_kind.eq('vs_neutral')&contrasts.left_persona.str.endswith('_explicit')&
        contrasts.marker_id.eq('religion_perspective_adoption')]
    selected.to_csv(output/'perspective_adoption_explicit_vs_neutral.csv',index=False)
    write_json(output/'manifest.json',dict(status='candidate_descriptive_summary',rows=len(frame),
        marker_occurrences=int(frame.marker_occurrences.sum()),new_marker_occurrences=int(frame.new_marker_occurrences.sum()),
        new_marker_ids=new,abstain_with_text=int(frame.abstain_with_text.sum()),
        ci_status_counts=contrasts.ci_status.value_counts().to_dict(),p_methods=contrasts.p_method.value_counts().to_dict(),
        report_manifest_sha256=sha256(report/'manifest.json'),script_sha256=sha256(__file__)))
    print(model.to_string(),flush=True)
    print(totals.loc[totals.added_in_v3].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();describe(args.report,args.output)
