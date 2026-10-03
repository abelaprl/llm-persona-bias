"""Combine computational shards into one whole-ETHICS candidate report."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import shutil
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import create_run_directory,write_json,sha256,KEYS

def concatenate_csv(paths,target):
    count = 0
    for path in paths:
        try:
            chunks = pd.read_csv(path,chunksize=10000)
            for chunk in chunks:
                chunk.to_csv(target,mode='w' if count==0 else 'a',header=count==0,index=False)
                count += len(chunk)
        except pd.errors.EmptyDataError:
            continue
    return count

def integrate(parts,output):
    parts = [Path(p) for p in parts]
    manifests = [json.loads((p/'manifest.json').read_text(encoding='utf-8')) for p in parts]
    if sum(m['rows'] for m in manifests)!=154428 or len({m['registry_version'] for m in manifests})!=1:
        raise ValueError('Incomplete or incompatible shards')
    source_manifests = [json.loads((p/'source_manifest.json').read_text(encoding='utf-8')) for p in parts]
    if any(m!=source_manifests[0] for m in source_manifests):
        raise ValueError('Shard source/code hashes differ')
    for path,info in source_manifests[0].items():
        if sha256(path)!=info['sha256']: raise ValueError('Input changed: '+path)
    output = create_run_directory(output)
    for sub in ('registry','response_level','occurrences','audit','inventory','statistics','comparison_with_baseline'):
        (output/sub).mkdir()
    for relative in ('source_manifest.json','registry/markers.json','statistics/design_manifest.json'):
        shutil.copyfile(parts[0]/relative,output/relative)
    shutil.copyfile(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_preservation_20260916_r1/manifest.json',output/'preservation_manifest.json')
    baseline_check=ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_baseline_identity_20260916_r1'
    shutil.copyfile(baseline_check/'manifest.json',output/'comparison_with_baseline/published_baseline_identity.json')
    shutil.copyfile(baseline_check/'column_comparison.csv',output/'comparison_with_baseline/published_baseline_columns.csv')
    effective_audit = ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_human_audit_20260916_r2'
    audit_design = json.loads((effective_audit/'manifest.json').read_text(encoding='utf-8'))
    write_json(output/'audit/design_manifest.json',dict(effective_package=str(effective_audit),
        design=audit_design,package_manifest_sha256=sha256(effective_audit/'manifest.json'),
        status='blank_templates_not_gold',supersedes_shard_audit_sampling=True))
    files = ['response_level/markers.csv.gz','occurrences/hits.csv.gz','occurrences/exclusions.csv.gz','inventory/coverage.csv',
        'statistics/overall_descriptive.csv','statistics/paired_availability.csv',
        'comparison_with_baseline/row_deltas.csv.gz','comparison_with_baseline/factorial_ablation.csv.gz']
    counts = {}
    for relative in files:
        counts[relative] = concatenate_csv([p/relative for p in parts],output/relative)
        print('Integrated '+relative+': '+str(counts[relative]),flush=True)
    keys = pd.read_csv(output/'response_level/markers.csv.gz',usecols=KEYS)
    if len(keys)!=154428 or keys.duplicated(KEYS).any() or keys.model_id.nunique()!=6 or keys.persona.nunique()!=17:
        raise ValueError('Integrated response universe is invalid')
    # One global stream avoids dependence on reused shard sampling streams.
    quotas = [pd.read_csv(p/'audit/rule_quotas.csv') for p in parts]
    from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,rule_origin
    all_rules = sorted(f'{m.marker_id}:{rule_origin(m.marker_id,i)}:{i}' for m in get_registry() for i,_ in enumerate(m.patterns))
    from src.benchmarks.ethics.religion_v3_precision_evidence import precision_evidence,write_precision_templates,sample_global_occurrences
    selected,summary=sample_global_occurrences(output/'occurrences/hits.csv.gz',all_rules)
    expected=pd.concat(quotas).groupby('rule_id').total_rule_hits.sum().reindex(all_rules,fill_value=0)
    assert summary.set_index('rule_id').total_rule_hits.eq(expected).all(),'Global occurrence counts disagree with shards'
    summary.to_csv(output/'audit/rule_quotas.csv',index=False)
    write_json(output/'audit/precision_sampling_manifest.json',dict(design='one global occurrence stream; per-rule reservoir of 100 or all if fewer',
        seed=18222008,supersedes_shard_samples=True,selection_independent_of_human_labels=True,sampled_occurrences=len(selected)))
    selected,evidence = precision_evidence(ROOT/'result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv',selected)
    selected.to_csv(output/'audit/precision_sample.csv',index=False)
    write_json(output/'audit/precision_full_responses.json',evidence)
    write_precision_templates(output,selected,evidence)
    # Keep source tables untouched; any unsupported equal-subset estimate is explicit.
    contrasts = pd.concat([pd.read_csv(p/'statistics/paired_contrasts.csv.gz') for p in parts],ignore_index=True)
    for column in ('effect','ci_low','ci_high','p_raw','p_holm'):
        contrasts[column+'_bootstrap_shard']=contrasts[column]
    incomplete = contrasts.population.eq('equal_subset_text') & contrasts[['n_commonsense','n_deontology','n_justice','n_virtue']].eq(0).any(axis=1)
    low_replication=contrasts.population.eq('equal_subset_text') & contrasts[['n_commonsense','n_deontology','n_justice','n_virtue']].lt(2).any(axis=1)
    contrasts['inference_scope_status'] = np.select([incomplete,contrasts.n_pairs.lt(2),low_replication],
        ['insufficient_four_subset_support','insufficient_pairs','insufficient_within_subset_replication'],default='available')
    contrasts.loc[incomplete,['effect','ci_low','ci_high','p_raw','p_holm']] = np.nan
    from src.benchmarks.ethics.religion_v3_support import augment_contrasts
    support_columns = KEYS+['word_count','explanation_available','final_output_available','surface_flags','answer_status']+[m.marker_id+'__count' for m in get_registry()]
    support_frame = pd.read_csv(output/'response_level/markers.csv.gz',usecols=support_columns)
    contrasts = augment_contrasts(support_frame,contrasts,[m.marker_id for m in get_registry()])
    del support_frame
    from src.benchmarks.ethics.religion_v3_inference import repair_inference,holm_planned
    contrasts=repair_inference(contrasts)
    contrasts['p_holm'] = contrasts.groupby(['model_id','population','metric','contrast_kind','family_id']).p_raw.transform(lambda p:holm_planned(p.to_numpy()))
    repaired_case=contrasts.loc[contrasts.model_id.eq('openai_gpt-oss-20b')&contrasts.population.eq('both_committed_text')&
        contrasts.left_persona.eq('jewish_implicit_2')&contrasts.right_persona.eq('jewish_explicit')&
        contrasts.marker_id.eq('connective_causal')&contrasts.metric.eq('prevalence_text')]
    assert len(repaired_case)==1 and repaired_case.n_pairs.iloc[0]==2 and repaired_case.p_raw.iloc[0]==.5
    repaired_case.to_csv(output/'statistics/degenerate_case_repair.csv',index=False)
    write_json(output/'statistics/inference_audit_manifest.json',dict(
        prevalence_test='exact paired McNemar for unweighted populations; paired sign-flip for equal-subset sensitivity',
        density_test='paired sign-flip of weighted mean differences; exact through 16 informative pairs, otherwise 10000 Monte Carlo draws with plus-one correction',
        randomization_assumption='within-question exchangeability of left and right observations under the null; two-sided absolute mean statistic',
        interval_policy='approximate question bootstrap CI; NA for degenerate resampling, fewer than six informative pairs, or insufficient stratum support; point estimates retained',
        correction='Holm across all planned rows in each registered family; unestimable p-values enter multiplicity as 1 and are displayed as NA',
        degenerate_intervals_removed=int(contrasts.bootstrap_degenerate.sum()),p_methods=contrasts.p_method.value_counts().to_dict(),
        p_underflow_floor=float(np.finfo(float).tiny),underflow_rows=int(contrasts.p_underflow_floor.sum()),
        original_shard_inference_retained_in_columns=True,
        motivating_case='GPT-OSS both_committed_text jewish_implicit_2 vs jewish_explicit connective_causal: two discordant pairs; exact p=0.5',
        sources=['https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html']))
    contrasts.to_csv(output/'statistics/paired_contrasts.csv.gz',index=False)
    counts['statistics/paired_contrasts.csv.gz'] = len(contrasts)
    from src.benchmarks.ethics.religion_lm_v3_diagnostics import sensitivity_directions
    sensitivity = sensitivity_directions(contrasts)
    sensitivity['direction_evaluable'] = np.isfinite(sensitivity.effect)&np.isfinite(sensitivity.effect_primary)
    sensitivity['direction_reversed'] = sensitivity.direction_reversed.astype('boolean')
    sensitivity.loc[~sensitivity.direction_evaluable,'direction_reversed'] = pd.NA
    sensitivity.to_csv(output/'statistics/sensitivity_directions.csv.gz',index=False)
    from src.benchmarks.ethics.religion_lm_v3_visuals import render_report
    import matplotlib.pyplot as plt
    plt.rcParams['figure.constrained_layout.h_pad']=.4
    figures = render_report(output)
    from src.benchmarks.ethics.build_religion_v3_explorer import build_explorer
    build_explorer(output)
    from src.benchmarks.ethics.render_religion_v3_publication import render_publication
    publication=render_publication(output)
    from src.benchmarks.ethics.render_religion_v3_density import render_density
    density_figure,default_notebook,notebook_path = render_density(output)
    figures.append(density_figure)
    counts_surface = {}
    for m in manifests:
        for k,v in m['surface_counts'].items(): counts_surface[k] = counts_surface.get(k,0)+v
    write_json(output/'manifest.json',dict(status='candidate_pending_two_independent_human_reviews',registry_version=manifests[0]['registry_version'],
        rows=154428,models=6,personas=17,marker_count=59,overall_scope='ethics_all',bootstrap_resamples=10000,seed=18222008,
        human_validation_complete=False,no_answer_scoring_changes=True,surface_counts=counts_surface,
        shards=[str(p) for p in parts],shard_manifest_sha256=[sha256(p/'manifest.json') for p in parts],
        file_rows=counts,figure_count=len(figures),integration_script_sha256=sha256(__file__),
        publication_panel_count=publication['count'],publication_pdf=publication['pdf'],
        default_notebook=default_notebook,notebook_path=notebook_path,
        equal_subset_insufficient_rows=int(incomplete.sum()),
        development_review=str(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_development_review_20260915_r1'),
        limitations=['upstream parsing remains candidate','independent human precision/recall pending','discovery saturation not reached','some function and referent attributes unresolved']))
    import platform
    from importlib.metadata import version
    write_json(output/'environment_manifest.json',dict(python=sys.version,platform=platform.platform(),
        packages={name:version(name) for name in ('numpy','pandas','scipy','matplotlib','nbformat')},
        executable=sys.executable,computation_backend='independent local processes; file exchange; no API'))
    from src.benchmarks.ethics.religion_lm_v3_io import build_source_manifest
    producer_names=['integrate_religion_v3_candidate.py','render_religion_v3_density.py','build_religion_v3_explorer.py',
        'religion_v3_support.py','religion_v3_precision_evidence.py','render_religion_v3_publication.py','religion_v3_inference.py','religion_v3_randomization.py']
    write_json(output/'integration_sources.json',build_source_manifest([ROOT/'src/benchmarks/ethics'/name for name in producer_names]))
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part',action='append',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    print(integrate(args.part,args.output),flush=True)

if __name__=='__main__': main()
