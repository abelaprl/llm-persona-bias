"""Move all rare-probe questions to development before their first readback."""
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import load_final_rows,create_run_directory,write_json,sha256
from src.benchmarks.ethics.religion_lm_v3_discovery import PROBES
from src.benchmarks.ethics.religion_lm_v3_audit import make_question_split,export_annotation_package
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface


def revise(previous, output):
    output = create_run_directory(output)
    source = ROOT/'result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv'
    rows = load_final_rows(source)
    old = pd.read_csv(previous/'question_split.csv')
    forced = set(old.loc[old.partition.eq('development'),['subset','question_id']].itertuples(index=False,name=None))
    probes = []
    for form_id,spec in PROBES.items():
        for _,group in rows.groupby('model_id'):
            hits = group.loc[group.final_output.str.contains(spec['pattern'],flags=re.I,regex=True)]
            if 0<len(hits)<10:
                hits = hits.copy()
                hits['form_id'] = form_id
                probes.append(hits)
                forced |= set(hits[['subset','question_id']].itertuples(index=False,name=None))
    split = make_question_split(rows,forced,18222008)
    split.to_csv(output/'question_split.csv',index=False)
    delta = old.merge(split,on=['subset','question_id'],suffixes=('_old','_new'),validate='one_to_one')
    delta.loc[delta.partition_old.ne(delta.partition_new)].to_csv(output/'partition_changes.csv',index=False)
    rare = pd.concat(probes).drop_duplicates('response_key').reset_index(drop=True)
    rare['readback_id'] = [f'Q{i+1:03d}' for i in range(len(rare))]
    rare['final_output'] = rare.final_output.map(lambda t:prepare_final_surface(t).visible_text)
    rare.to_json(output/'rare_full_responses.json',orient='records',force_ascii=False,indent=2)
    audit = export_annotation_package(rows,split)
    audit.drop(columns='final_output').to_csv(output/'human_recall_key.csv',index=False)
    blind = audit[['audit_id','final_output']].copy()
    blind['final_output'] = blind.final_output.map(lambda t:prepare_final_surface(t).visible_text)
    blind.to_json(output/'human_recall_blind.json',orient='records',force_ascii=False,indent=2)
    write_json(output/'manifest.json',dict(status='candidate_partition_revision',source_sha256=sha256(source),previous=str(previous),
        reason='Read all rare-probe hits while keeping their complete question families out of independent evaluation.',
        rare_responses=len(rare),partition_counts=split.partition.value_counts().to_dict(),human_recall_responses=len(audit),
        supersedes_human_recall_package=str(previous/'human_recall_blind.json'),human_review_status='not_started'))
    print(f'Revision frozen; {len(rare)} rare responses',flush=True)


if __name__=='__main__':
    revise(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_preparation_20260915_r1',ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_partition_revision_20260915_r1')
