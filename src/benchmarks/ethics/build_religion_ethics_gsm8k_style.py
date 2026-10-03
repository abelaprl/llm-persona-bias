"""Build a new immutable presentation bundle, then optionally install it."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from src.benchmarks.ethics.religion_ethics_gsm8k_style_data import (
    SOURCE, NOTEBOOK, MODELS, CONTRAST_KEY, LABELS, load_data, select_examples,
)
from src.benchmarks.ethics.religion_ethics_gsm8k_style_plots import render_all
from src.benchmarks.ethics.religion_ethics_gsm8k_style_notebook import build_notebook, execute_notebook, rebase_notebook


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def source_files():
    relative = ['manifest.json', 'response_level/markers.csv.gz', 'statistics/overall_descriptive.csv',
                'statistics/paired_contrasts.csv.gz', 'statistics/paired_availability.csv',
                'occurrences/hits.csv.gz', 'registry/markers.json', 'inventory/coverage.csv']
    files = [SOURCE / name for name in relative]
    files += sorted((SOURCE / 'figures').rglob('*.png')) + sorted((SOURCE / 'figures').rglob('*.svg'))
    files += [ROOT / 'notebooks/religion/linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb',
              ROOT / 'result/benchmarks/ethics/reports/religion_lm_v3_preparation_20260915_r1/targeted_full_responses.json',
              ROOT / 'result/benchmarks/ethics/reports/religion_lm_v3_preparation_20260915_r1/open_discovery_round_1.json']
    return files


def verify_tables(output, source_contrasts):
    index = pd.read_csv(output / 'figure_index.csv')
    checked = 0
    source = source_contrasts.set_index(CONTRAST_KEY + ['population'])
    for record in index.itertuples():
        table = pd.read_csv(output / record.table)
        if {'effect', 'ci_low', 'ci_high', 'p_holm', 'population'}.issubset(table.columns):
            keys = pd.MultiIndex.from_frame(table[CONTRAST_KEY + ['population']])
            expected = source.loc[keys].reset_index()
            for column in ['effect', 'ci_low', 'ci_high', 'p_holm', 'n_pairs', 'raw_count_left', 'raw_count_right']:
                np.testing.assert_allclose(table[column], expected[column], rtol=1e-12, atol=1e-12, equal_nan=True,
                                           err_msg=f'{record.figure_id}: {column}')
            checked += len(table)
        with Image.open(output / record.png) as image:
            image.verify()
    return dict(figures=len(index), source_contrast_rows_checked=checked,
                png_valid=True, effect_ci_holm_n_raw_counts_match_source=True)


def build(run_id, output):
    if output.exists():
        raise FileExistsError(f'Run already exists; choose a new run ID: {output}')
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in source_files()}
    old_notebook_hash = sha256(NOTEBOOK)
    data = load_data()
    if len(data['rows']) != 154428 or len(data['registry']) != 59:
        raise ValueError('Unexpected v3 source scope')
    output.mkdir(parents=True, exist_ok=False)
    tables = output / 'tables'
    tables.mkdir()
    data['coverage'].to_csv(tables / 'coverage.csv', index=False)
    data['summary'].to_csv(tables / 'overall_descriptive.csv', index=False)
    aligned = data['summary'][data['summary'].marker_id.eq('religion_identity_' + data['summary'].identity)]
    aligned.to_csv(tables / 'aligned_identity.csv', index=False)
    data['terms'].to_csv(tables / 'religious_terms_by_model.csv', index=False)
    data['availability'].to_csv(tables / 'availability.csv', index=False)
    data['sensitivity'].to_csv(tables / 'sensitivity_pairs.csv.gz', index=False)
    sensitivity_summary = data['sensitivity'].groupby(['population', 'metric']).agg(
        registered=('evaluable', 'size'), evaluable=('evaluable', 'sum'), direction_reversed=('direction_reversed', 'sum')).reset_index()
    sensitivity_summary.to_csv(tables / 'sensitivity_summary.csv', index=False)
    registry = pd.DataFrame(data['registry'])
    registry['label'] = registry.marker_id.map(LABELS)
    registry[['marker_id', 'label', 'family_id', 'operational_definition', 'theoretical_source',
              'exclusion_rule', 'validation_status', 'registry_version']].to_csv(tables / 'registry.csv', index=False)
    shutil.copy2(SOURCE / 'statistics/paired_contrasts.csv.gz', tables / 'paired_contrasts.csv.gz')
    example_directory = ROOT / 'result/benchmarks/ethics/reports/religion_lm_v3_preparation_20260915_r1'
    examples = select_examples({str((example_directory / name).relative_to(ROOT)).replace('\\', '/'):
                                json.loads((example_directory / name).read_text(encoding='utf-8'))
                                for name in ['targeted_full_responses.json', 'open_discovery_round_1.json']})
    write_json(output / 'examples.json', examples)
    print('[3/5] Membuat grafik bergaya GSM8K', flush=True)
    figures = render_all(output, data)
    code_directory = output / 'code_snapshot'
    code_directory.mkdir()
    for name in ['religion_ethics_gsm8k_style_data.py', 'religion_ethics_gsm8k_style_plots.py',
                 'religion_ethics_gsm8k_style_notebook.py', 'build_religion_ethics_gsm8k_style.py']:
        shutil.copy2(Path(__file__).with_name(name), code_directory / name)
    copied_test = ROOT / 'src/benchmarks/ethics/tests/test_religion_ethics_gsm8k_style.py'
    shutil.copy2(copied_test, code_directory / copied_test.name)
    artifact_hashes = {str(p.relative_to(output)).replace('\\', '/'): sha256(p)
                       for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = dict(status='candidate_pending_independent_human_validation',
                    source_bundle=str(SOURCE.relative_to(ROOT)).replace('\\', '/'), run_id=run_id,
                    presentation_reference='notebooks/religion/linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb',
                    responses=len(data['rows']), markers=len(data['registry']), models=6, personas=17,
                    figures=len(figures), hidden_reasoning_used=False, scoring_changed=False,
                    inference_recomputed=False, source_hashes=hashes, artifact_hashes=artifact_hashes,
                    previous_active_notebook_sha256=old_notebook_hash,
                    local_figures=f'notebooks/religion/linguistic marker/ETHICS/figures/gsm8k_style_{run_id}',
                    limitations=['candidate registry', 'human precision/recall pending',
                                 'text-conditional effects require availability context',
                                 'stance markers are multi-label, not opening composition',
                                 'density effects are not rank-biserial',
                                 'term plots contain existing lexical spans, not a new validated lexicon'])
    write_json(output / 'manifest.json', manifest)
    print('[4/5] Mengeksekusi sel notebook dan menyimpan output', flush=True)
    notebook = execute_notebook(build_notebook(run_id))
    with (output / 'religion_ethics_gsm8k_style.workspace.ipynb').open('x', encoding='utf-8') as handle:
        nbformat.write(notebook, handle)
    report_notebook = rebase_notebook(notebook, NOTEBOOK.parent, output, ROOT)
    with (output / 'religion_ethics_gsm8k_style.ipynb').open('x', encoding='utf-8') as handle:
        nbformat.write(report_notebook, handle)
    print('[5/5] Memeriksa angka, gambar, dan integritas sumber', flush=True)
    verification = verify_tables(output, data['contrasts'])
    for relative, expected in hashes.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Source changed: {relative}')
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    images = sum('image/png' in out.get('data', {}) for c in code_cells for out in c.outputs)
    if images != len(figures):
        raise ValueError(f'Notebook did not embed all figures: {images} / {len(figures)}')
    verification.update(source_files_unchanged=len(hashes), notebook_code_cells_executed=len(code_cells),
                        notebook_embedded_png=images, notebook_sha256=sha256(output / 'religion_ethics_gsm8k_style.workspace.ipynb'),
                        report_notebook_sha256=sha256(output / 'religion_ethics_gsm8k_style.ipynb'))
    write_json(output / 'verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


def archive_notebook(run_id):
    archive = NOTEBOOK.parent / 'archive' / f'before_gsm8k_style_{run_id}'
    archive.mkdir(parents=True, exist_ok=False)
    notebook = rebase_notebook(nbformat.read(NOTEBOOK, as_version=4), NOTEBOOK.parent, archive, ROOT)
    archived_path = archive / NOTEBOOK.name
    with archived_path.open('x', encoding='utf-8') as handle:
        nbformat.write(notebook, handle)
    write_json(archive / 'manifest.json', dict(original_path=str(NOTEBOOK), original_sha256=sha256(NOTEBOOK),
                                            archived_sha256=sha256(archived_path), only_navigation_paths_adjusted=True))
    return archived_path


def install(run_id, output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    if sha256(NOTEBOOK) != manifest['previous_active_notebook_sha256']:
        raise ValueError('Active notebook changed after build; refusing to replace it')
    notebook_source = output / 'religion_ethics_gsm8k_style.workspace.ipynb'
    if sha256(notebook_source) != verification['notebook_sha256']:
        raise ValueError('Built notebook changed')
    target = NOTEBOOK.parent / 'figures' / f'gsm8k_style_{run_id}'
    if target.exists():
        raise FileExistsError(target)
    for relative, expected in manifest['artifact_hashes'].items():
        if sha256(output / relative) != expected:
            raise ValueError('Artifact changed: ' + relative)
    shutil.copytree(output / 'figures', target)
    archived = archive_notebook(run_id)
    shutil.copy2(notebook_source, NOTEBOOK)
    for path in target.iterdir():
        if sha256(path) != sha256(output / 'figures' / path.name):
            raise ValueError('Installed figure differs: ' + path.name)
    result = dict(active_notebook=str(NOTEBOOK), archive=str(archived), figures=str(target),
                  png_count=len(list(target.glob('*.png'))), svg_count=len(list(target.glob('*.svg'))),
                  notebook_sha256=sha256(NOTEBOOK), all_figure_hashes_match=True)
    write_json(output / 'installation_verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--install', action='store_true', help='Install an already verified bundle; archive the active notebook first.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('run-id may contain only letters, digits, underscores and hyphens')
    output = ROOT / 'result/benchmarks/ethics/reports' / f'religion_ethics_gsm8k_style_{args.run_id}'
    if args.install:
        install(args.run_id, output)
    else:
        build(args.run_id, output)


if __name__ == '__main__':
    main()
