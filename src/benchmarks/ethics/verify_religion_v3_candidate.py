"""Audit candidate output identities, numerators, figures and source hashes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import re
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import nbformat
from src.benchmarks.ethics.religion_lm_v3_io import KEYS,sha256,write_json,create_run_directory
from src.benchmarks.ethics.religion_lm_v3_registry import get_registry,FAMILIES,family_for
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final

def verify(report,output):
    output = create_run_directory(output)
    numeric_types={m.marker_id+'__'+suffix:float for m in get_registry() for suffix in ('count','density','present','present_output')}
    frame = pd.read_csv(report/'response_level/markers.csv.gz',dtype=numeric_types,low_memory=False)
    manifest = json.loads((report/'manifest.json').read_text(encoding='utf-8'))
    assert len(frame)==154428 and not frame.duplicated(KEYS).any()
    assert frame.model_id.nunique()==6 and frame.persona.nunique()==17 and frame.subset.nunique()==4
    assert not any('ground_truth' in c or c.startswith('accuracy') for c in frame)
    source = json.loads((report/'source_manifest.json').read_text(encoding='utf-8'))
    checks = {path:sha256(path)==info['sha256'] for path,info in source.items()}
    assert all(checks.values()),'A source changed'
    auxiliary=json.loads((report/'integration_sources.json').read_text(encoding='utf-8'))
    auxiliary_checks={path:sha256(path)==info['sha256'] for path,info in auxiliary.items()}
    assert all(auxiliary_checks.values()),'An integration source changed'
    preservation=json.loads((report/'preservation_manifest.json').read_text(encoding='utf-8'))
    preservation_checks={path:sha256(path)==info['sha256'] for path,info in preservation['files'].items()}
    assert all(preservation_checks.values()),'A preserved old presentation asset changed'
    baseline_identity=json.loads((report/'comparison_with_baseline/published_baseline_identity.json').read_text(encoding='utf-8'))
    assert baseline_identity['rows']==154428 and baseline_identity['mismatched_cells']==0
    assert all(sha256(path)==info['sha256'] for path,info in baseline_identity['sources'].items())
    lexical = frame.explanation_available
    final = frame.final_output_available
    markers = [m.marker_id for m in get_registry()]
    for marker in markers:
        count,density,present = [frame[marker+'__'+suffix] for suffix in ('count','density','present')]
        assert density.loc[~lexical].isna().all() and present.loc[~lexical].isna().all()
        assert np.allclose(density.loc[lexical],100*count.loc[lexical]/frame.loc[lexical,'word_count'])
        assert present.loc[lexical].eq(count.loc[lexical].gt(0).astype(int)).all()
        assert frame.loc[~final,marker+'__present_output'].isna().all()
        assert frame.loc[final,marker+'__present_output'].eq(count.loc[final].gt(0).astype(int)).all()
        assert count.loc[~final].isna().all()
    for family in FAMILIES:
        member_cols = [m+'__present' for m in markers if family_for(m)==family]
        assert frame.loc[lexical,family+'__family_present'].eq(frame.loc[lexical,member_cols].gt(0).any(axis=1).astype(int)).all()
    # Every accepted occurrence contributes once to the published count.
    observed = Counter()
    for chunk in pd.read_csv(report/'occurrences/hits.csv.gz',usecols=['model_id','persona','marker_id','raw_start','raw_end','candidate_status'],chunksize=50000):
        assert not chunk.candidate_status.eq('excluded').any()
        assert chunk.raw_start.lt(chunk.raw_end).all()
        observed.update(chunk.groupby(['model_id','persona','marker_id']).size().to_dict())
    for (model,persona),group in frame.groupby(['model_id','persona']):
        for marker in markers:
            assert int(group[marker+'__count'].sum())==observed[(model,persona,marker)]
    comparisons = pd.read_csv(report/'statistics/paired_contrasts.csv.gz')
    assert comparisons.subset.eq('ethics_all').all()
    assert comparisons.bootstrap_resamples.eq(10000).all()
    assert comparisons.groupby('model_id').apply(lambda g:g[['contrast_kind','left_persona','right_persona']].drop_duplicates().shape[0],include_groups=False).eq(52).all()
    assert comparisons.n_pairs.le(comparisons.total_pairs).all()
    assert comparisons.p_holm.dropna().between(0,1).all()
    assert comparisons.p_raw.dropna().gt(0).all()
    assert set(comparisons.p_method.unique())<={'exact_paired_McNemar','exact_paired_sign_flip','monte_carlo_paired_sign_flip','not_estimated_insufficient_support'}
    assert comparisons.loc[comparisons.bootstrap_degenerate,['ci_low','ci_high']].isna().all().all()
    assert comparisons.loc[comparisons.ci_status.ne('available'),['ci_low','ci_high']].isna().all().all()
    from src.benchmarks.ethics.religion_v3_inference import holm_planned
    recomputed=comparisons.groupby(['model_id','population','metric','contrast_kind','family_id']).p_raw.transform(lambda p:holm_planned(p.to_numpy()))
    assert np.allclose(comparisons.p_holm,recomputed,equal_nan=True)
    assert (comparisons.ci_low.le(comparisons.ci_high)|comparisons.ci_low.isna()).all()
    finite=comparisons.effect.notna()
    assert np.allclose(comparisons.loc[finite,'effect'],(comparisons.estimate_left-comparisons.estimate_right).loc[finite])
    assert comparisons[['raw_count_left','raw_count_right']].ge(0).all().all()
    summary = pd.read_csv(report/'statistics/overall_descriptive.csv')
    assert len(summary)==6*17*59 and summary.subset.eq('ethics_all').all()
    grouped = frame.groupby(['model_id','persona'])
    expected_prevalence = grouped[[m+'__present' for m in markers]].mean()*100
    expected_total = grouped.size()
    expected_available = grouped.explanation_available.sum()
    for _,r in summary.iterrows():
        key = (r.model_id,r.persona)
        assert r.total==expected_total.loc[key] and r.linguistic_available==expected_available.loc[key]
        assert np.isclose(r.prevalence_text,expected_prevalence.loc[key,r.marker_id+'__present'],equal_nan=True)
    figures = json.loads((report/'figure_manifest.json').read_text(encoding='utf-8'))
    figures += json.loads((report/'figure_manifest_additional.json').read_text(encoding='utf-8'))
    covered = set()
    pair_keys=['model_id','population','metric','contrast_kind','left_persona','right_persona','marker_id']
    figure_numeric=['effect','ci_low','ci_high','p_holm','n_pairs','raw_count_left','raw_count_right']
    pair_reference=comparisons.set_index(pair_keys)[figure_numeric]
    descriptor_keys=['model_id','persona','marker_id']
    descriptor_numeric=['total','final_available','linguistic_available','raw_count','word_total','prevalence_text','prevalence_output','mean_response_density']
    descriptor_reference=summary.set_index(descriptor_keys)[descriptor_numeric]
    for figure in figures:
        assert figure['scope']=='ethics_all'
        for path,expected in figure['sha256'].items(): assert sha256(report/path)==expected
        assert sha256(report/figure['table'])==figure['table_sha256']
        table = pd.read_csv(report/figure['table'])
        assert table.subset.eq('ethics_all').all()
        if figure['figure_id']=='01_availability':
            availability_keys=['model_id','persona','surface_status']
            reference_counts=frame.groupby(availability_keys).size()
            selected=table.set_index(availability_keys)
            assert selected.total.eq(reference_counts.loc[selected.index]).all()
            assert np.allclose(table.percent,100*table.total/table.groupby(['model_id','persona']).total.transform('sum'))
        elif '_atlas_' in figure['figure_id']:
            selected=table.set_index(descriptor_keys)
            assert np.allclose(selected[descriptor_numeric],descriptor_reference.loc[selected.index],equal_nan=True)
        else:
            selected=table.set_index(pair_keys)
            assert np.allclose(selected[figure_numeric],pair_reference.loc[selected.index],equal_nan=True)
        covered.update(figure['outcome_ids'])
    assert set(markers)<=covered
    publication=json.loads((report/'publication_panel_manifest.json').read_text(encoding='utf-8'))
    assert publication['count']==24 and not publication['preview']
    assert sha256(report/publication['pdf'])==publication['pdf_sha256']
    assert publication['source_sha256']==sha256(report/'statistics/paired_contrasts.csv.gz')
    numeric=['effect','ci_low','ci_high','n_pairs','raw_count_left','raw_count_right']
    reference=comparisons.set_index(pair_keys)[numeric]
    for panel in publication['panels']:
        assert panel['scope']=='ethics_all' and panel['layout_check_passed'] and not panel['preview']
        for path,expected in panel['sha256'].items():assert sha256(report/path)==expected
        assert sha256(report/panel['table'])==panel['table_sha256']
        table=pd.read_csv(report/panel['table'])
        assert table.subset.eq('ethics_all').all()
        selected=table.set_index(pair_keys)
        assert np.allclose(selected[numeric],reference.loc[selected.index,numeric],equal_nan=True)
    notebook = nbformat.read(report/manifest['default_notebook'],as_version=4)
    nbformat.validate(notebook)
    notebook_links=0
    notebook_paths=[report/manifest['default_notebook'],Path(manifest['notebook_path'])]
    if (report/'presentation_review_manifest.json').exists():
        presentation=json.loads((report/'presentation_review_manifest.json').read_text(encoding='utf-8'))
        assert sha256(Path(presentation['workspace_notebook']))==presentation['workspace_notebook_sha256']
        assert sha256(report/presentation['preferred_report_notebook'])==presentation['report_notebook_sha256']
        notebook_paths.append(report/presentation['preferred_report_notebook'])
    for path in notebook_paths:
        document=nbformat.read(path,as_version=4);nbformat.validate(document)
        for cell in document.cells:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if link.startswith(('https://','http://','#')):continue
                target=(path.parent/link).resolve()
                assert target.is_file(),'Broken notebook link: '+str(target)
                notebook_links+=1
    explorer=json.loads((report/'explorer_manifest.json').read_text(encoding='utf-8'))
    assert explorer['rows']==len(comparisons) and sha256(report/explorer['path'])==explorer['sha256']
    design = json.loads((report/'audit/design_manifest.json').read_text(encoding='utf-8'))
    package = Path(design['effective_package'])
    audit_key = pd.read_csv(package/'sampling_design_key.csv')
    split = pd.read_csv(package/'question_split.csv')
    assert len(audit_key)==1020 and audit_key.question_id.nunique()>10 and audit_key.subset.nunique()==4
    assert audit_key.merge(split,on=['subset','question_id'],suffixes=('','_effective')).partition_effective.eq('held_out').all()
    readback = pd.read_csv(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_development_review_20260915_r1/readback_index.csv')
    assert readback.merge(split,on=['subset','question_id']).partition.eq('development').all()
    rules = pd.read_csv(report/'audit/rule_quotas.csv')
    assert rules.human_reviewed.eq(0).all() and rules.false_positives.isna().all()
    result = dict(status='automated_checks_passed_human_validation_pending',rows=len(frame),models=6,personas=17,subsets=4,markers=59,
        figures=len(figures),source_hashes_unchanged=checks,accepted_occurrences=int(sum(observed.values())),
        resolved_notebook_file_links=notebook_links,inference_repair_checks_passed=True,
        publication_panels=publication['count'],publication_layout_and_numeric_checks_passed=True,
        integration_hashes_unchanged=auxiliary_checks,all_contrasts_explorer_rows=explorer['rows'],
        preserved_old_asset_hashes_unchanged=preservation_checks,
        published_baseline_comparison_columns=baseline_identity['compared_columns'],published_baseline_mismatched_cells=0,
        contrasts=len(comparisons),human_recall_responses=len(audit_key),human_recall_question_clusters=audit_key.question_id.nunique(),
        full_discovery_readback_rows=len(readback),no_development_heldout_overlap=True,checks_not_precision_or_recall=True,
        verification_script_sha256=sha256(__file__),report_manifest_sha256=sha256(report/'manifest.json'))
    write_json(output/'verification.json',result)
    compact={k:(len(v) if isinstance(v,dict) else v) for k,v in result.items()}
    print(json.dumps(compact,ensure_ascii=False,default=str),flush=True)

def main():
    parser = argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.report,args.output)

if __name__=='__main__':main()
