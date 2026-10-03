"""Export independent held-out hit audit, keeping global diagnostic audit intact."""
import argparse
import gzip
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory,write_json,sha256
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,rule_origin
from src.benchmarks.ethics.religion_v3_precision_evidence import sample_global_occurrences,precision_evidence,write_precision_templates


def package(report,output):
    split_path=ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_human_audit_20260916_r2/question_split.csv'
    split=pd.read_csv(split_path,dtype=str)
    eligible=set(split.loc[split.partition.eq('held_out'),['subset','question_id']].itertuples(index=False,name=None))
    if len(eligible)!=297:raise ValueError('Unexpected independent question split')
    output=create_run_directory(output);(output/'audit').mkdir()
    filtered=output/'heldout_occurrences.csv.gz';total=0
    with gzip.open(filtered,'xt',encoding='utf-8',newline='') as handle:
        for chunk in pd.read_csv(report/'occurrences/hits.csv.gz',chunksize=50000,keep_default_na=False):
            mask=[key in eligible for key in chunk[['subset','question_id']].itertuples(index=False,name=None)]
            chosen=chunk.loc[mask]
            if len(chosen):chosen.to_csv(handle,index=False,header=total==0);total+=len(chosen)
    rules=sorted(f'{m.marker_id}:{rule_origin(m.marker_id,i)}:{i}' for m in get_registry() for i,_ in enumerate(m.patterns))
    selected,quotas=sample_global_occurrences(filtered,rules)
    assert set(selected[['subset','question_id']].itertuples(index=False,name=None))<=eligible
    source=ROOT/'result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv'
    selected,evidence=precision_evidence(source,selected)
    selected['partition']='held_out'
    selected.to_csv(output/'audit/precision_sample.csv',index=False)
    quotas.to_csv(output/'audit/rule_quotas.csv',index=False)
    write_json(output/'audit/precision_full_responses.json',evidence)
    write_precision_templates(output,selected,evidence)
    write_json(output/'manifest.json',dict(status='blank_independent_human_precision_templates_not_gold',
        eligible_question_clusters=len(eligible),eligible_occurrences=total,sampled_occurrences=len(selected),
        sampled_responses=len(evidence),rules=len(quotas),rules_without_heldout_hits=int(quotas.total_rule_hits.eq(0).sum()),
        development_and_calibration_excluded=True,seed=18222008,human_reviews_completed=0,
        source_sha256=sha256(source),split_sha256=sha256(split_path),occurrences_sha256=sha256(report/'occurrences/hits.csv.gz'),
        script_sha256=sha256(__file__),purpose='independent held-out rule precision audit; global report reservoir remains diagnostic'))
    print('Held-out precision package complete: '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();package(args.report,args.output)
