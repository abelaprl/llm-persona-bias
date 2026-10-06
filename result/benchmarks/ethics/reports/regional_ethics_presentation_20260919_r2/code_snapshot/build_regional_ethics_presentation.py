"""Build a new immutable Regional presentation bundle; no external API."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import pandas as pd
from .regional_ethics_presentation_data import ROOT, SOURCE, POLICY, PROVENANCE, load_provenance, load_bundle, regional_marker_ids, sha256, write_json
from .regional_ethics_presentation_view import build_view, coverage_table
from .regional_ethics_expression_data import profile_table, diversity_rows, diversity_tables
from .regional_ethics_presentation_stats import build_statistics, engine


def assert_pairing_ready():
    # User approved exclusion from paired inference only; preserve all source rows.
    return {r['response_key'] for r in load_provenance()['scenario_overrides']}


def create_run(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def prepare_tables(bundle, staging):
    tables = staging / 'tables'
    tables.mkdir(exist_ok=True)
    print('[view] validating surfaces and eligibility', flush=True)
    view = build_view(bundle, bundle.source / 'surface_texts.jsonl.gz')
    view.to_csv(tables / 'analysis_view.csv.gz', index=False, compression='gzip')
    view.loc[~view.pairing_eligible, ['response_key', 'model_id', 'persona', 'subset', 'question_id', 'explanation_available', 'pairing_exclusion_reason']].to_csv(tables / 'pairing_exclusions.csv', index=False)
    shutil.copy2(PROVENANCE, tables / 'source_provenance_audit.json')
    bundle.catalog.to_csv(tables / 'marker_catalog.csv', index=False)
    coverage_table(view).to_csv(tables / 'coverage.csv', index=False)
    view.groupby(['model_id', 'persona', 'text_status', 'analysis_surface_status']).size().rename('n').reset_index().to_csv(tables / 'eligibility_changes.csv', index=False)
    print(view.analysis_surface_status.value_counts().to_dict(), flush=True)
    print('[profiles] all 53 markers and 23-category diversity', flush=True)
    profiles = profile_table(view, bundle.catalog.marker_id.tolist())
    profiles.to_csv(tables / 'all_marker_profiles.csv', index=False)
    profiles[profiles.marker_id.isin(regional_marker_ids(bundle.catalog))].to_csv(tables / 'regional_profiles.csv', index=False)
    diversity = diversity_rows(view, regional_marker_ids(bundle.catalog))
    diversity.to_csv(tables / 'diversity_response_level.csv.gz', index=False, compression='gzip')
    for name, table in diversity_tables(diversity).items():
        table.to_csv(tables / f'{name}.csv', index=False)
    scoring = view.copy()
    scoring['is_correct'] = engine.nullable_bool(scoring.is_correct)
    scoring['numeric_semantic_conflict'] = engine.bool_series(scoring.numeric_semantic_conflict)
    scoring['candidate_uncommitted'] = scoring.analysis_answer_status.ne('committed') & scoring.candidate_answer.notna()
    engine.build_scoring_summary(scoring).to_csv(tables / 'scoring_per_subset.csv', index=False)
    build_statistics(view, bundle.catalog, tables / 'statistics')
    return view


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for filename, expected in manifest['input_hashes'].items():
        if sha256(filename) != expected:
            raise ValueError(f'Input changed: {filename}')
    for filename, expected in manifest.get('artifact_hashes', {}).items():
        path = (bundle / filename).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != expected:
            raise ValueError(f'Artifact mismatch: {filename}')
    tables = bundle / 'tables'
    coverage = pd.read_csv(tables / 'coverage.csv')
    profiles = pd.read_csv(tables / 'regional_profiles.csv')
    stats = pd.read_csv(tables / 'statistics/paired_contrasts.csv.gz')
    if coverage.total.sum() != 154428 or len(profiles) != 2346:
        raise ValueError('Incomplete response/profile coverage')
    populations = stats.population.value_counts().to_dict()
    if populations != {'text_all': 17472, 'text_both_committed': 7056}:
        raise ValueError(f'Unexpected comparison counts: {populations}')
    index = pd.read_csv(bundle / 'figure_index.csv')
    if index.figure_id.duplicated().any():
        raise ValueError('Duplicate figure IDs')
    from .religion_ethics_print import audit_svg
    for row in index.itertuples(index=False):
        for column in ['png', 'svg', 'data_file']:
            if not (bundle / getattr(row, column)).is_file():
                raise ValueError('Missing figure or data table')
        audit = audit_svg(bundle / row.svg)
        if not np.isclose(audit['width_cm'], 16) or audit['height_cm'] > 26.001 or audit['min_font_pt'] < 8.5:
            raise ValueError(f'Print audit failed: {row.figure_id}')
    from .regional_ethics_presentation_notebook import validate_notebook
    notebook_info = validate_notebook(bundle / 'linguistic_marker_regional_cultural_ethics_v3_candidate.ipynb', bundle)
    return dict(valid=True, status='candidate', total=154428, n_text=int(coverage.n_text.sum()),
                regional_markers=23, profiles=len(profiles), populations=populations, figures=len(index),
                input_hashes_checked=len(manifest['input_hashes']), **notebook_info)


def build(run_id, source=SOURCE, stage='all'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    output = ROOT / f'result/ETHICS/reports/regional_ethics_presentation_{run_id}'
    if stage == 'verify':
        print(json.dumps(verify_bundle(output), indent=2), flush=True)
        return output
    assert_pairing_ready()
    if output.exists():
        raise FileExistsError(output)
    staging = ROOT / f'.test_tmp/regional_ethics_presentation_{run_id}'
    bundle = load_bundle(source)
    inputs = dict(input_hashes=bundle.hashes, policy=POLICY, run_id=run_id)
    if staging.exists():
        if json.loads((staging / 'staging_inputs.json').read_text(encoding='utf-8')) != inputs:
            raise ValueError('Staging input/policy mismatch')
    else:
        create_run(staging)
        write_json(staging / 'staging_inputs.json', inputs)
    view = prepare_tables(bundle, staging)
    from .regional_ethics_expression_evidence import build_evidence
    print('[evidence] literal context and matched response cards', flush=True)
    build_evidence(bundle, view, staging / 'evidence')
    from .regional_ethics_presentation_plots import render_all, load_tables
    render_all(load_tables(staging), staging)
    from .regional_ethics_presentation_notebook import make_notebook, execute_notebook
    import nbformat
    notebook = staging / 'linguistic_marker_regional_cultural_ethics_v3_candidate.ipynb'
    nbformat.write(make_notebook(staging, staging, published_bundle=output), notebook)
    execute_notebook(notebook, ROOT)
    codes = sorted(Path(__file__).parent.glob('*regional_ethics*.py'))
    code_dir = staging / 'code_snapshot'
    code_dir.mkdir(exist_ok=True)
    for path in codes:
        shutil.copy2(path, code_dir / path.name)
    manifest = dict(**inputs, source=str(bundle.source), result_status='candidate', human_review_complete=False,
                    release_gate=bundle.release_gate, published_bundle=str(output),
                    code_hashes={str(p): sha256(p) for p in codes},
                    versions={p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'scipy', 'matplotlib', 'nbformat']})
    write_json(staging / 'manifest.json', manifest)
    verification = verify_bundle(staging)
    write_json(staging / 'verification.json', verification)
    manifest['artifact_hashes'] = {str(p.relative_to(staging)): sha256(p) for p in staging.rglob('*') if p.is_file() and p != staging / 'manifest.json'}
    write_json(staging / 'manifest.json', manifest)
    verify_bundle(staging)
    if output.exists() or not output.resolve().is_relative_to((ROOT / 'result/ETHICS/reports').resolve()):
        raise ValueError('Unsafe publication target')
    staging.rename(output)
    print(f'[published] {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--stage', choices=['all', 'verify'], default='all')
    args = parser.parse_args()
    build(args.run_id, args.source, args.stage)
