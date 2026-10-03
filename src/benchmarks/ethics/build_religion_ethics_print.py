"""Build immutable, print-sized presentation results; install only after checks."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import inspect
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import pandas as pd
from src.benchmarks.ethics.religion_ethics_gsm8k_style_data import SOURCE, NOTEBOOK
from src.benchmarks.ethics.religion_ethics_gsm8k_style_plots import render_all
from src.benchmarks.ethics.religion_ethics_gsm8k_style_notebook import execute_notebook, rebase_notebook
from src.benchmarks.ethics.build_religion_ethics_gsm8k_style import sha256, write_json, verify_tables
from src.benchmarks.ethics.build_religion_ethics_expression import notebook_links
from src.benchmarks.ethics.religion_ethics_expression import RELIGION_MARKERS
from src.benchmarks.ethics.religion_ethics_print import (
    PrintRenderer, diversity_rows, diversity_tables, render_expression_print, render_diversity, audit_svg,
)

BASE = ROOT / 'result/benchmarks/ethics/reports/religion_ethics_gsm8k_style_20260918_r2'
EXPRESSION = ROOT / 'result/benchmarks/ethics/reports/religion_ethics_expression_20260919_r1'


def choose_figure(local, bundle, expected):
    if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == expected:
        return local
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected:
        raise ValueError('Gambar tidak cocok dengan manifest cetak: ' + str(bundle))
    return bundle


def validate_main_cache(index, source_index):
    if set(index.parent_id) != set(source_index.figure_id) or index.figure_id.duplicated().any():
        raise ValueError('Incomplete or ambiguous main-render cache')
    for source in source_index.itertuples():
        count = {'coverage': 1, 'sensitivity': 6, 'direct_identity': 6,
                 'implicit_vs_explicit': 3}.get(source.section, 4)
        group = index[index.parent_id.eq(source.figure_id)].sort_values('page')
        pages = list(range(1, (count + 1) // 2 + 1))
        panels = [min(2, count - 2 * (p - 1)) for p in pages]
        names = [source.figure_id + f'_p{p:02d}' for p in pages]
        if (group.page.tolist() != pages or group.panels.tolist() != panels
                or group.figure_id.tolist() != names or not group.section.eq(source.section).all()):
            raise ValueError('Missing, reordered or inconsistent cached pages: ' + source.figure_id)


def prepare_print_notebook(original, bundle_relative):
    if 'print_layout' in original.metadata:
        raise ValueError('Use the preserved pre-print base notebook for reproduction')
    result = copy.deepcopy(original)
    setup = next(c for c in result.cells if c.cell_type == 'code' and 'def show_section(section):' in c.source)
    setup.source += f'''

# The scientific table sources above remain unchanged.
PRINT_BUNDLE = ROOT / {bundle_relative!r}
print_manifest = json.loads((PRINT_BUNDLE / 'manifest.json').read_text(encoding='utf-8'))
print_index = pd.read_csv(PRINT_BUNDLE / 'figure_index.csv')

{inspect.getsource(choose_figure)}

def show_section(section):
    for record in print_index[print_index.section.eq(section)].itertuples():
        local = ACTIVE_DIR / 'figures' / Path(record.png).name
        expected = print_manifest['artifact_hashes'][record.png]
        path = choose_figure(local, PRINT_BUNDLE / record.png, expected)
        display(Image(filename=str(path), width=900))
        display(Markdown(record.caption))
        links = []
        for target, label in [(path, 'PNG 300 dpi'), (path.with_suffix('.svg'), 'SVG 16 cm'),
                              (PRINT_BUNDLE / record.table, 'CSV lengkap')]:
            relative = os.path.relpath(target, DOCUMENT_DIR).replace(chr(92), '/')
            links.append('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>')
        display(HTML(' · '.join(links)))
'''
    expression = next(c for c in result.cells if c.cell_type == 'code' and 'def show_expression(section):' in c.source)
    expression.source = re.sub(r'def show_expression\(section\):.*?(?=\nshow_expression\()',
                               'show_expression = show_section\n', expression.source, flags=re.S)
    for cell in result.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## 5B.'):
            cell.source = cell.source.replace('Empat heatmap', 'Empat kelompok heatmap (dua model per halaman)')
        if cell.cell_type == 'code':
            if 'build_religion_ethics_gsm8k_style.py' in cell.source:
                cell.source = cell.source.replace('build_religion_ethics_gsm8k_style.py', 'build_religion_ethics_print.py')
                cell.source = cell.source.replace('--run-id 20260918_reproduction_01',
                                                  '--run-id print_reproduction_01 --base-notebook ' + bundle_relative + '/base_notebook.ipynb')
                cell.source = cell.source.replace("show_table('figure_index.csv', ['figure_id', 'title', 'section', 'rows'], limit=40)",
                                                  "display(print_index[['figure_id', 'title', 'section', 'rows']])")
            cell.outputs = []
            cell.execution_count = None
    position = next(i for i,c in enumerate(result.cells) if c.cell_type == 'markdown' and c.source.startswith('## 6.'))
    result.cells[position:position] = [nbformat.v4.new_markdown_cell('''## 5D. Keragaman kategori marker per respons

Jumlah jenis marker berbeda dihitung dari **20 kategori kamus kandidat** yang sama, bukan banyaknya kata unik atau seluruh kekayaan bahasa. Pengulangan kategori tidak menambah keragaman. Kurva kumulatif menunjukkan persentase respons dengan jumlah jenis ≤ nilai pada sumbu x; kurva lebih ke kanan berarti jumlah jenis terdeteksi lebih banyak.

Semua respons berteks disertakan, termasuk abstain dan nol hit. Respons tanpa teks penjelasan tetap tercatat dalam total dan availability, tetapi keragamannya NA, bukan nol. Tidak ada pengujian signifikansi atau klaim sebab-akibat baru. Panjang respons dapat meningkatkan jumlah jenis yang ditemukan: CSV pendamping memisahkan rentang 1–49, 50–99, 100–199, 200–399, dan ≥400 kata. Stratifikasi kasar ini bukan pengendalian statistik lengkap; panjang juga dapat berbeda di dalam rentang.

Count mentah menjumlahkan occurrence antar-marker (overlap kategori dapat dihitung lebih dari sekali); density adalah rata-rata count per 100 kata pada tiap respons. **Jumlah jenis tidak dibagi 100 kata dan tidak disebut density.**'''),
        nbformat.v4.new_code_cell('''show_section('diversity')
for filename, label in [('diversity_summary.csv', 'Keragaman, count, density dan coverage'),
                        ('diversity_by_length.csv', 'Ringkasan deskriptif menurut panjang respons'),
                        ('diversity_distribution.csv', 'Distribusi lengkap termasuk nol'),
                        ('diversity_responses.csv.gz', 'Satu baris per respons; tanpa teks = NA')]:
    path = PRINT_BUNDLE / 'tables' / filename
    relative = os.path.relpath(path, DOCUMENT_DIR).replace(chr(92), '/')
    display(HTML('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>'))
display(pd.read_csv(PRINT_BUNDLE / 'tables/diversity_summary.csv').head(17))
''')]
    result.metadata['print_layout'] = dict(width_cm=16, min_font_pt=8.5, png_dpi=300,
                                          bundle=bundle_relative, detector_changed=False)
    return result


def build(run_id, output, base_notebook=NOTEBOOK, main_cache=None):
    if output.exists():
        raise FileExistsError('Choose a new run ID; existing results are immutable')
    original = nbformat.read(base_notebook, as_version=4)
    if 'print_layout' in original.metadata:
        raise ValueError('Pass --base-notebook pointing to the pre-print base snapshot')
    hashes = {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p)
              for bundle in [BASE, EXPRESSION] for p in bundle.rglob('*') if p.is_file()}
    for name in ['registry/markers.json', 'response_level/markers.csv.gz', 'statistics/paired_contrasts.csv.gz']:
        path = SOURCE / name
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    previous_notebook = sha256(NOTEBOOK)
    previous_figures = {p.name: sha256(p) for p in (NOTEBOOK.parent / 'figures').iterdir()
                        if p.is_file() and p.suffix in ['.png','.svg']}
    temporary_root = ROOT / '.test_tmp'
    temporary_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='religion_print_', dir=temporary_root))
    print('STAGING:', stage, flush=True)
    data = {key: pd.read_csv(BASE / 'tables' / filename) for key, filename in [
        ('coverage','coverage.csv'), ('summary','overall_descriptive.csv'),
        ('terms','religious_terms_by_model.csv'), ('contrasts','paired_contrasts.csv.gz'),
        ('availability','availability.csv'), ('sensitivity','sensitivity_pairs.csv.gz')]}
    print('[1/5] Menata ulang grafik tanpa menghitung ulang inferensi', flush=True)
    if main_cache is None:
        main_index = render_all(stage, data, renderer_class=PrintRenderer)
    else:
        # Reuse only the completed main-render stage; all tables and fonts are
        # checked below against the same immutable source before publication.
        main_index = pd.read_csv(main_cache / 'figure_index.csv')
        validate_main_cache(main_index, pd.read_csv(BASE / 'figure_index.csv'))
        (stage / 'figures').mkdir()
        (stage / 'figure_tables').mkdir()
        for relative in set(main_index.png) | set(main_index.svg) | set(main_index.table):
            source = (main_cache / relative).resolve()
            if not source.is_relative_to(main_cache.resolve()):
                raise ValueError('Cache path escapes the requested directory')
            shutil.copy2(source, stage / relative)
        print(f'  Memakai {len(main_index)} halaman utama; validasi penuh tetap dijalankan.', flush=True)
    profiles = pd.read_csv(EXPRESSION / 'tables/marker_profiles.csv')
    cards = json.loads((EXPRESSION / 'response_cards.json').read_text(encoding='utf-8'))
    extra = render_expression_print(stage, profiles, cards)
    print('[2/5] Menghitung keragaman kategori pada seluruh respons', flush=True)
    columns = ['response_key','model_id','persona','subset','question_id','surface_status','answer_status','word_count']
    columns += [m + '__count' for m in RELIGION_MARKERS]
    rows = pd.read_csv(SOURCE / 'response_level/markers.csv.gz', usecols=columns)
    if len(rows) != 154428:
        raise ValueError('Unexpected frozen response scope')
    derived = diversity_rows(rows, RELIGION_MARKERS)
    summary, distribution, lengths = diversity_tables(derived)
    tables = stage / 'tables'
    tables.mkdir()
    derived.to_csv(tables / 'diversity_responses.csv.gz', index=False)
    summary.to_csv(tables / 'diversity_summary.csv', index=False)
    distribution.to_csv(tables / 'diversity_distribution.csv', index=False)
    lengths.to_csv(tables / 'diversity_by_length.csv', index=False)
    extra += render_diversity(stage, distribution)
    index = pd.concat([main_index, pd.DataFrame(extra)], ignore_index=True)
    index.to_csv(stage / 'figure_index.csv', index=False)
    print('[3/5] Memeriksa tabel, font, ukuran, dan sumber beku', flush=True)
    verification = verify_tables(stage, data['contrasts'])
    compared = 0
    for path in (BASE / 'figure_tables').glob('*.csv'):
        actual = pd.read_csv(stage / 'figure_tables' / path.name)
        expected = pd.read_csv(path)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
        compared += len(expected)
    profile_pages = pd.concat([pd.read_csv(stage / r.table) for r in index.itertuples()
                               if r.section == 'expression_heatmaps'])
    keys = ['model_id','persona','marker_id']
    pd.testing.assert_frame_equal(profile_pages.drop_duplicates(keys).sort_values(keys).reset_index(drop=True),
                                  profiles.sort_values(keys).reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    for record in index[index.section.eq('expression_cards')].itertuples():
        actual = pd.read_csv(stage / record.table)
        expected = pd.read_csv(EXPRESSION / 'figure_tables' / (record.parent_id + '.csv'))
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_exact=False)
    typography = {r.figure_id: audit_svg(stage / r.svg) for r in index.itertuples()}
    for identifier, audit in typography.items():
        if abs(audit['width_cm'] - 16) > 1e-5 or audit['height_cm'] > 26 or audit['min_font_pt'] < 8.5 or audit['font_families'] != ['DejaVu Sans']:
            raise ValueError(f'Typography mismatch: {identifier}: {audit}')
    for relative, digest in hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    snapshots = stage / 'code_snapshot'
    snapshots.mkdir()
    for filename in ['religion_ethics_print.py','build_religion_ethics_print.py',
                     'religion_ethics_gsm8k_style_plots.py','religion_ethics_gsm8k_style_data.py',
                     'religion_ethics_expression.py','religion_ethics_expression_plots.py',
                     'religion_ethics_gsm8k_style_notebook.py']:
        shutil.copy2(Path(__file__).with_name(filename), snapshots / filename)
    shutil.copy2(Path(__file__).parent / 'tests/test_religion_ethics_print.py', snapshots / 'test_religion_ethics_print.py')
    with (stage / 'base_notebook.ipynb').open('x', encoding='utf-8') as handle:
        nbformat.write(original, handle)
    manifest = dict(run_id=run_id, status='candidate_pending_independent_human_validation',
                    source_bundle=SOURCE.relative_to(ROOT).as_posix(), responses=len(rows), markers=59,
                    religion_markers=20, figures=len(index), width_cm=16, min_font_pt=8.5, png_dpi=300,
                    detector_changed=False, scoring_changed=False, inference_recomputed=False,
                    diversity_is_descriptive=True, full_dataset=True, source_hashes=hashes,
                    previous_active_notebook_sha256=previous_notebook, previous_figure_hashes=previous_figures,
                    artifact_hashes={p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob('*') if p.is_file()},
                    limitations=['category diversity is dictionary-dependent and length-dependent',
                                 'length bins are descriptive, not causal adjustment',
                                 'scope and independent human precision/recall remain unresolved'])
    write_json(stage / 'manifest.json', manifest)
    verification.update(original_figure_table_rows_unchanged=compared, profile_cells_unchanged=len(profiles),
                        card_responses_unchanged=len(cards), source_files_unchanged=len(hashes),
                        diversity_response_rows=len(derived), typography=typography)
    # Publish to a new location only after all render and data checks pass.
    shutil.copytree(stage, output)
    print('[4/5] Mengeksekusi notebook dengan seluruh gambar baru', flush=True)
    notebook = execute_notebook(prepare_print_notebook(original, output.relative_to(ROOT).as_posix()))
    image_count = sum('image/png' in out.get('data', {}) for c in notebook.cells for out in c.get('outputs', []))
    if image_count != len(index):
        raise ValueError(f'Embedded figure mismatch: {image_count} != {len(index)}')
    verification.update(notebook_code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                        notebook_embedded_png=image_count, notebook_links=notebook_links(notebook, NOTEBOOK.parent))
    with (output / 'print.workspace.ipynb').open('x', encoding='utf-8') as handle:
        nbformat.write(notebook, handle)
    report = rebase_notebook(notebook, NOTEBOOK.parent, output, ROOT)
    with (output / 'print.ipynb').open('x', encoding='utf-8') as handle:
        nbformat.write(report, handle)
    verification['notebook_sha256'] = sha256(output / 'print.workspace.ipynb')
    write_json(output / 'verification.json', verification)
    print('[5/5] Build tervalidasi; instalasi terpisah setelah pemeriksaan visual', flush=True)
    print(json.dumps({k:v for k,v in verification.items() if k != 'typography'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--base-notebook', type=Path, default=NOTEBOOK)
    parser.add_argument('--main-cache', type=Path, help='Reuse a completed main-render stage; verify all tables and typography again.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    output = ROOT / 'result/benchmarks/ethics/reports' / ('religion_ethics_print_' + args.run_id)
    build(args.run_id, output, args.base_notebook, args.main_cache)


if __name__ == '__main__':
    main()
