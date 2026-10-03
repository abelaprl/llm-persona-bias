"""Freeze inputs, surfaces, partitions and unlabelled review packages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.benchmarks.ethics.religion_lm_v3_io import load_final_rows, create_run_directory, write_json, build_source_manifest
from src.benchmarks.ethics.religion_lm_v3_audit import make_question_split, export_annotation_package
from src.benchmarks.ethics.religion_lm_v3_discovery import scan_observed_forms, targeted_readback, build_discovery_queue
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
import pandas as pd


def prepare(source, output, seed=18222008):
    output = create_run_directory(output)
    samples_path = ROOT/'result/benchmarks/ethics/reports/religion_final_response_plan_audit_20260915/full_final_samples.json'
    baseline = ROOT/'result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
    sources = [source,samples_path,baseline]
    sources += sorted((ROOT/'src/core/linguistic_markers').glob('*.py'))
    write_json(output/'source_manifest.json',build_source_manifest(sources))
    print('Loading final-only source',flush=True)
    rows = load_final_rows(source)
    samples = json.loads(samples_path.read_text(encoding='utf-8'))
    forced = {(r['subset'],r['question_id']) for r in samples}
    split = make_question_split(rows,forced,seed)
    split.to_csv(output/'question_split.csv',index=False)
    scan_observed_forms(rows).to_csv(output/'form_inventory.csv',index=False)
    targeted = targeted_readback(rows,split,seed)
    targeted.drop(columns='final_output').to_csv(output/'targeted_readback_index.csv',index=False)
    unique = targeted.drop_duplicates('response_key').copy().reset_index(drop=True)
    unique['readback_id'] = [f'D{i+1:03d}' for i in range(len(unique))]
    # Raw source has already been delimiter-screened; serialize only the visible final surface.
    unique['final_output'] = unique.final_output.map(lambda t:prepare_final_surface(t).visible_text)
    unique.to_json(output/'targeted_full_responses.json',orient='records',force_ascii=False,indent=2)
    old_keys = {json.dumps([r['model_id'],r['persona'],r['subset'],r['question_id']],separators=(',',':')) for r in samples}
    prior = old_keys | set(unique.response_key)
    for round_id in (1,2):
        queue = build_discovery_queue(rows,split,prior,{'per_model':10},seed+round_id)
        prior |= set(queue.response_key)
        queue = queue.copy()
        queue['final_output'] = queue.final_output.map(lambda t:prepare_final_surface(t).visible_text)
        queue.to_json(output/f'open_discovery_round_{round_id}.json',orient='records',force_ascii=False,indent=2)
    audit = export_annotation_package(rows,split,seed=seed)
    audit.drop(columns='final_output').to_csv(output/'human_recall_key.csv',index=False)
    blind = audit[['audit_id','final_output']].copy()
    blind['final_output'] = blind.final_output.map(lambda t:prepare_final_surface(t).visible_text)
    blind.to_json(output/'human_recall_blind.json',orient='records',force_ascii=False,indent=2)
    print(f'Partition frozen; {len(unique)} targeted full responses; {len(audit)} blind human audit responses',flush=True)
    surface_rows = []
    for i,row in enumerate(rows.itertuples(index=False)):
        s = prepare_final_surface(row.final_output)
        surface_rows.append(dict(response_key=row.response_key,model_id=row.model_id,persona=row.persona,subset=row.subset,question_id=row.question_id,
            answer_status=row.answer_status,surface_status=s.status,final_output_available=s.final_output_available,explanation_available=s.explanation_available,
            word_count=s.word_count,sentence_count=s.sentence_count,ttr_300=s.ttr_300,surface_flags='|'.join(s.flags)))
        if (i+1)%25000==0:
            print(f'Surfaces {i+1}/{len(rows)}',flush=True)
    surfaces = pd.DataFrame(surface_rows)
    surfaces.to_csv(output/'surface_inventory.csv.gz',index=False)
    counts = surfaces.groupby(['model_id','persona','subset','answer_status','surface_status'],dropna=False).size().rename('total').reset_index()
    counts.to_csv(output/'surface_counts.csv',index=False)
    write_json(output/'preparation_manifest.json',{'status':'candidate_preparation_only','rows':len(rows),'seed':seed,
        'targeted_unique_responses':len(unique),'targeted_selections':len(targeted),'open_discovery_responses':120,
        'human_recall_responses':len(audit),'human_review_status':'not_started','rule_promotion':'none',
        'word_denominator':'Unicode tokens containing at least one alphabetic character; technical digits excluded',
        'partition_counts':split.partition.value_counts().to_dict(),'surface_counts':surfaces.surface_status.value_counts().to_dict()})
    print('Preparation complete',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.source,args.output)
