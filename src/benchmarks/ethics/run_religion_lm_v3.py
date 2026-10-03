"""Rebuild final-only Religion ETHICS candidate outputs without overwriting runs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import random
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import load_final_rows,create_run_directory,write_json,build_source_manifest,KEYS
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,registry_rows,VERSION
from src.benchmarks.ethics.religion_lm_v3_audit import promote_rules


@lru_cache(maxsize=256)
def cached_extract(text):
    return extract_final(text,ablation=True)

def extract_batch(records):
    results = []
    for record in records:
        cached_row,cached_hits = cached_extract(record['final_output'])
        row,hits = dict(cached_row),[dict(h) for h in cached_hits]
        metadata = {k:record[k] for k in KEYS+['response_key','answer_status']}
        row.update(metadata)
        for hit in hits:
            hit.update(metadata)
        results.append((row,hits))
    return results


def verify_audit_bundle(path):
    if path is None:
        raise ValueError('Validated runs require independently annotated human audit data; use --stage candidates for preparation.')
    path = Path(path)
    if not (path/'rule_validation_reports.json').is_file():
        raise ValueError('Missing human rule validation reports')
    reports = json.loads((path/'rule_validation_reports.json').read_text(encoding='utf-8'))
    if not reports or not all(promote_rules(report)['approved'] for report in reports):
        raise ValueError('Human validation gate rejected this bundle')
    # Gold-file evaluation and registry/hash agreement are mandatory, not a manual status toggle.
    if not (path/'gold_annotations.csv').is_file() or not (path/'evaluation_manifest.json').is_file():
        raise ValueError('Missing evaluated gold evidence')
    return reports


def run_pipeline(source,output,seed=18222008,workers=1,resamples=10000,expected_rows=154428,render=True,only_models=None):
    output = create_run_directory(output)
    for sub in ('registry','response_level','occurrences','audit','inventory','statistics','comparison_with_baseline'):
        (output/sub).mkdir()
    baseline = ROOT/'result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
    paths = [source,baseline] + sorted((ROOT/'src/benchmarks/ethics').glob('*religion_lm_v3*.py'))
    paths += sorted((ROOT/'src/core/linguistic_markers').glob('*.py'))
    write_json(output/'source_manifest.json',build_source_manifest(paths))
    write_json(output/'registry/markers.json',registry_rows())
    partition = ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_partition_revision_20260915_r1'
    write_json(output/'audit/design_manifest.json',dict(effective_partition=str(partition),
        hashes=build_source_manifest([partition/'question_split.csv',partition/'human_recall_key.csv',partition/'human_recall_blind.json']),
        human_reviewers_completed=0,precision_sample_unit='rule occurrence; repeated response clusters must be retained in audit',
        recall_unit='response-marker plus raw-final spans; independent annotations required'))
    write_json(output/'statistics/design_manifest.json',dict(scope='ethics_all',unit='response',pairing=['model_id','subset','question_id'],
        effect_direction='left minus right',bootstrap_resamples=resamples,bootstrap_unit='question',
        correction='Holm within model/population/metric/contrast_kind/family',
        all_results_exploratory_until_human_validation=True,baseline_ablation='2x2 surface-and-scope bundle versus registry; replay baseline separately checked',
        forest_fixed_outcomes=['religion_perspective_adoption','religion_moral_authority_framing','religion_named_concept_reference','religion_identity_irrelevance','ai_identity_statement','response_binary_decline']))
    rows = load_final_rows(source)
    if only_models is not None:
        rows = rows.loc[rows.model_id.isin(only_models)].copy()
    if expected_rows is not None and len(rows)!=expected_rows:
        raise ValueError(f'Expected {expected_rows} responses, got {len(rows)}')
    records = rows.to_dict('records')
    batches = [records[i:i+250] for i in range(0,len(records),250)]
    rng = random.Random(seed)
    rule_counts = Counter()
    reservoir = defaultdict(list)
    results = []
    writers = {}
    handles = {kind:gzip.open(output/f'occurrences/{kind}.csv.gz','wt',encoding='utf-8',newline='') for kind in ('hits','exclusions')}
    ablation_handle = gzip.open(output/'comparison_with_baseline/factorial_ablation.csv.gz','wt',encoding='utf-8',newline='')
    ablation_writer = None
    pool = ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    iterator = pool.map(extract_batch,batches,buffersize=workers*2) if pool else map(extract_batch,batches)
    try:
        for batch in iterator:
            for row,hits in batch:
                ablation_columns = [c for c in row if c.startswith('ablation_')]
                ablation_row = {k:row[k] for k in KEYS+['word_count','explanation_available']+[m.marker_id+'__count' for m in get_registry()]+ablation_columns}
                if ablation_writer is None:
                    ablation_writer = csv.DictWriter(ablation_handle,fieldnames=list(ablation_row));ablation_writer.writeheader()
                ablation_writer.writerow(ablation_row)
                for c in ablation_columns: del row[c]
                results.append(row)
                for hit in hits:
                    kind = 'exclusions' if hit['candidate_status']=='excluded' else 'hits'
                    if kind not in writers:
                        writers[kind] = csv.DictWriter(handles[kind],fieldnames=list(hit))
                        writers[kind].writeheader()
                    writers[kind].writerow(hit)
                    if kind=='hits':
                        key = hit['rule_id']
                        rule_counts[key] += 1
                        bucket = reservoir[key]
                        if len(bucket)<100:
                            bucket.append(hit)
                        else:
                            index = rng.randrange(rule_counts[key])
                            if index<100:
                                bucket[index] = hit
            if len(results)%5000==0 or len(results)==len(rows):
                print(f'Extracted {len(results)}/{len(rows)}',flush=True)
    finally:
        for handle in handles.values():
            handle.close()
        ablation_handle.close()
        if pool:
            pool.shutdown()
    frame = pd.DataFrame(results)
    del results,records,batches
    frame.to_csv(output/'response_level/markers.csv.gz',index=False)
    quota = pd.DataFrame([dict(rule_id=k,total_rule_hits=v,sampled_occurrences=len(reservoir[k]),human_reviewed=0,false_positives=None,status='pending_human_review') for k,v in rule_counts.items()])
    quota.to_csv(output/'audit/rule_quotas.csv',index=False)
    pd.DataFrame([r for bucket in reservoir.values() for r in bucket]).to_csv(output/'audit/precision_sample.csv',index=False)
    selected_keys = {r['response_key'] for bucket in reservoir.values() for r in bucket}
    evidence = rows.loc[rows.response_key.isin(selected_keys),['response_key','final_output']].copy()
    from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
    evidence['final_output'] = evidence.final_output.map(lambda t:prepare_final_surface(t).visible_text)
    evidence.to_json(output/'audit/precision_full_responses.json',orient='records',force_ascii=False,indent=2)
    inventory = frame.groupby(['model_id','persona','subset','answer_status','surface_status'],dropna=False).size().rename('total').reset_index()
    inventory.to_csv(output/'inventory/coverage.csv',index=False)
    previous = pd.read_csv(baseline,usecols=lambda c:c in KEYS+['text_available','word_count'] or c.endswith('__count'))
    common = [m.marker_id+'__count' for m in get_registry() if m.marker_id+'__count' in previous]
    joined = frame.merge(previous[KEYS+['text_available','word_count']+common],on=KEYS,suffixes=('_new','_old'),validate='one_to_one')
    delta = joined[KEYS].copy()
    delta['old_text_available'] = joined.text_available
    delta['new_explanation_available'] = joined.explanation_available
    delta['word_count_delta'] = joined.word_count_new-joined.word_count_old
    for c in common:
        delta[c+'_delta'] = joined[c+'_new'].fillna(0)-joined[c+'_old'].fillna(0)
    delta.to_csv(output/'comparison_with_baseline/row_deltas.csv.gz',index=False)
    from src.benchmarks.ethics.religion_lm_v3_statistics import summarize_overall,estimate_contrasts
    summary = summarize_overall(frame)
    summary.to_csv(output/'statistics/overall_descriptive.csv',index=False)
    from src.benchmarks.ethics.religion_lm_v3_diagnostics import availability_diagnostics,sensitivity_directions
    availability_diagnostics(frame).to_csv(output/'statistics/paired_availability.csv',index=False)
    if workers>1:
        groups = [(g,seed,resamples) for _,g in frame.groupby('model_id',sort=True)]
        with ProcessPoolExecutor(max_workers=min(workers,len(groups))) as stats_pool:
            comparisons = pd.concat(stats_pool.map(statistics_batch,groups),ignore_index=True)
    else:
        comparisons = estimate_contrasts(frame,seed,resamples)
    comparisons.to_csv(output/'statistics/paired_contrasts.csv.gz',index=False)
    sensitivity_directions(comparisons).to_csv(output/'statistics/sensitivity_directions.csv.gz',index=False)
    if render:
        from src.benchmarks.ethics.religion_lm_v3_visuals import render_report
        render_report(output)
    write_json(output/'manifest.json',dict(status='candidate_pending_two_independent_human_reviews',registry_version=VERSION,
        rows=len(frame),models=frame.model_id.nunique(),personas=frame.persona.nunique(),seed=seed,
        bootstrap_resamples=resamples,marker_count=len(get_registry()),comparison_rows=len(comparisons),
        surface_counts=frame.surface_status.value_counts().to_dict(),human_validation_complete=False,
        overall_scope='ethics_all',no_answer_scoring_changes=True,
        multiplicity_family=['model_id','population','metric','contrast_kind','family_id']))
    return output

def statistics_batch(args):
    from src.benchmarks.ethics.religion_lm_v3_statistics import estimate_contrasts
    return estimate_contrasts(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv')
    parser.add_argument('--stage',choices=('candidates','all'),default='candidates')
    parser.add_argument('--audit-bundle',type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--seed',type=int,default=18222008)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--model',action='append',dest='models')
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--bootstrap-resamples',type=int,default=10000)
    args = parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():
        parser.error('Run ID must contain only letters, digits, underscores and hyphens')
    if args.stage=='all':
        verify_audit_bundle(args.audit_bundle)
    output = ROOT/f'result/benchmarks/ethics/reports/religion_lm_v3_candidate_{args.run_id}'
    print(run_pipeline(args.source,output,args.seed,args.workers,args.bootstrap_resamples,
        expected_rows=25738*len(args.models) if args.models else 154428,render=not args.no_render,only_models=args.models),flush=True)


if __name__=='__main__':
    main()
