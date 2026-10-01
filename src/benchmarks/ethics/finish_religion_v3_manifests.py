"""Finish missing manifests after notebook permission failure; never rerender or overwrite."""
import argparse
import json
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import pandas as pd
from importlib.metadata import version
from src.benchmarks.ethics.religion_lm_v3_io import write_json,sha256,build_source_manifest


def finish(output):
    targets=['manifest.json','environment_manifest.json','integration_sources.json','completion_recovery.json']
    for name in targets:
        if (output/name).exists():raise FileExistsError(output/name)
    base=ROOT/'result/ETHICS/reports'
    parts=[base/f'religion_lm_v3_candidate_20260915_r2_shard_{s}' for s in 'abc']
    manifests=[json.loads((p/'manifest.json').read_text(encoding='utf-8')) for p in parts]
    figures=json.loads((output/'figure_manifest.json').read_text(encoding='utf-8'))
    figures+=json.loads((output/'figure_manifest_additional.json').read_text(encoding='utf-8'))
    publication=json.loads((output/'publication_panel_manifest.json').read_text(encoding='utf-8'))
    for figure in figures+publication['panels']:
        for path,expected in figure['sha256'].items():assert sha256(output/path)==expected
        assert sha256(output/figure['table'])==figure['table_sha256']
    assert len(figures)==13 and publication['count']==24
    assert sha256(output/publication['pdf'])==publication['pdf_sha256']
    notebook_name='religion_ethics_linguistic_markers_v3_presentation_candidate.ipynb'
    notebook_path=ROOT/'notebooks/religion'/('linguistic_marker_religion_ethics_v3_candidate_'+output.name.removeprefix('religion_lm_v3_candidate_')+'.ipynb')
    assert notebook_path.is_file() and (output/notebook_name).is_file()
    for path,info in json.loads((output/'source_manifest.json').read_text(encoding='utf-8')).items():assert sha256(path)==info['sha256']
    contrast=pd.read_csv(output/'statistics/paired_contrasts.csv.gz',usecols=['inference_scope_status'])
    quotas=pd.read_csv(output/'audit/rule_quotas.csv')
    counts={'response_level/markers.csv.gz':154428,'occurrences/hits.csv.gz':int(quotas.total_rule_hits.sum()),
        'occurrences/exclusions.csv.gz':56485,'inventory/coverage.csv':1273,'statistics/overall_descriptive.csv':6018,
        'statistics/paired_availability.csv':192,'comparison_with_baseline/row_deltas.csv.gz':154428,
        'comparison_with_baseline/factorial_ablation.csv.gz':154428,'statistics/paired_contrasts.csv.gz':len(contrast)}
    assert counts['occurrences/hits.csv.gz']==2865351 and len(contrast)==165672
    surface={}
    for manifest in manifests:
        for key,value in manifest['surface_counts'].items():surface[key]=surface.get(key,0)+value
    write_json(output/'manifest.json',dict(status='candidate_pending_two_independent_human_reviews',registry_version=manifests[0]['registry_version'],
        rows=154428,models=6,personas=17,marker_count=59,overall_scope='ethics_all',bootstrap_resamples=10000,seed=18222008,
        human_validation_complete=False,no_answer_scoring_changes=True,surface_counts=surface,shards=[str(p) for p in parts],
        shard_manifest_sha256=[sha256(p/'manifest.json') for p in parts],file_rows=counts,figure_count=len(figures),
        integration_script_sha256=sha256(ROOT/'src/benchmarks/ethics/integrate_religion_v3_candidate.py'),
        publication_panel_count=publication['count'],publication_pdf=publication['pdf'],default_notebook=notebook_name,notebook_path=str(notebook_path),
        equal_subset_insufficient_rows=int(contrast.inference_scope_status.eq('insufficient_four_subset_support').sum()),
        development_review=str(base/'religion_lm_v3_development_review_20260915_r1'),
        limitations=['upstream parsing remains candidate','independent human precision/recall pending','discovery saturation not reached','some function and referent attributes unresolved'],
        completion_recovery='Notebook write required approved filesystem escalation; all report artifacts already existed; only missing manifests added'))
    write_json(output/'environment_manifest.json',dict(python=sys.version,platform=platform.platform(),
        packages={name:version(name) for name in ('numpy','pandas','scipy','matplotlib','nbformat')},executable=sys.executable,
        computation_backend='independent local processes; file exchange; no API'))
    names=['integrate_religion_v3_candidate.py','render_religion_v3_density.py','build_religion_v3_explorer.py',
        'religion_v3_support.py','religion_v3_precision_evidence.py','render_religion_v3_publication.py','religion_v3_inference.py','religion_v3_randomization.py',
        'copy_religion_v3_notebook.py','finish_religion_v3_manifests.py']
    write_json(output/'integration_sources.json',build_source_manifest([ROOT/'src/benchmarks/ethics'/name for name in names]))
    write_json(output/'completion_recovery.json',dict(reason='PermissionError creating workspace notebook after all report graphics and notebook were written',
        no_existing_report_file_modified=True,figures_checked=13,publication_panels_checked=24,notebook_sha256=sha256(notebook_path),
        count_provenance='completed integration stdout and global rule-quotas reconciliation; full verification follows',script_sha256=sha256(__file__)))
    print('Missing manifests completed without overwriting report artifacts.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();finish(args.report)
    from src.benchmarks.ethics.verify_religion_v3_candidate import verify
    from src.benchmarks.ethics.summarize_religion_v3_candidate import summarize
    base=ROOT/'result/ETHICS/reports'
    verify(args.report,base/'religion_lm_v3_verification_20260916_r4')
    summarize(args.report,base/'religion_lm_v3_summary_20260916_r4')
