"""Add frozen religion-marker profiles and paired response cards to the notebook."""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import json
import os
import re
import shutil
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from src.benchmarks.ethics.build_religion_ethics_gsm8k_style import sha256, write_json
from src.benchmarks.ethics.religion_ethics_gsm8k_style_data import SOURCE, NOTEBOOK, MODELS
from src.benchmarks.ethics.religion_ethics_gsm8k_style_notebook import execute_notebook, rebase_notebook
from src.benchmarks.ethics.religion_ethics_expression import (
    RELIGION_MARKERS, CARD_IDENTITIES, heatmap_table, select_card_rows, highlight_segments,
)
from src.benchmarks.ethics.religion_ethics_expression_plots import render_expression

RAW_SOURCE = ROOT / 'result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv'
KEYS = ['model_id', 'persona', 'subset', 'question_id']


def extend_notebook(notebook, run_id):
    result = copy.deepcopy(notebook)
    if 'expression_supplement' in result.metadata:
        raise ValueError('An expression supplement is already installed; refusing to duplicate it')
    cells = [nbformat.v4.new_markdown_cell('''## 5B. Profil lengkap 20 marker bahasa agama

Empat heatmap menampilkan seluruh 20 marker agama pada enam model dan skenario N/E/I1/I2/I3. Warna memakai **prevalence respons berteks (%)** dengan skala identik 0–100, bukan jumlah istilah unik atau skor kekayaan bahasa. N yang sama diulang pada setiap identitas. Termasuk abstain berteks; respons kosong dan label-only tetap ada dalam total tetapi tidak masuk denominator teks. Nol dibedakan dari NA.

Tabel setiap gambar menyimpan count mentah, denominator, dan rata-rata density per respons per 100 kata. Nama marker adalah definisi operasional **kandidat**, bukan bukti keyakinan atau fungsi wacana yang telah tervalidasi. Grafik ini deskriptif; effect size, CI dan koreksi Holm tetap dibaca dari bagian inferensial notebook.'''),
             nbformat.v4.new_code_cell(f'''SUPPLEMENT = ROOT / 'result/ETHICS/reports/religion_ethics_expression_{run_id}'
SUPPLEMENT_LOCAL = ACTIVE_DIR / 'figures/expression_{run_id}'
supplement_manifest = json.loads((SUPPLEMENT / 'manifest.json').read_text(encoding='utf-8'))
supplement_index = pd.read_csv(SUPPLEMENT / 'figure_index.csv')

def show_expression(section):
    for record in supplement_index[supplement_index.section.eq(section)].itertuples():
        local = SUPPLEMENT_LOCAL / Path(record.png).name
        path = local if local.is_file() else SUPPLEMENT / record.png
        if hashlib.sha256(path.read_bytes()).hexdigest() != supplement_manifest['artifact_hashes'][record.png]:
            raise ValueError('Supplement figure does not match manifest: ' + str(path))
        display(Image(filename=str(path), width=1600))
        display(Markdown(record.caption))
        links = []
        for target, label in [(path, 'PNG'), (path.with_suffix('.svg'), 'SVG'), (SUPPLEMENT / record.table, 'CSV lengkap')]:
            relative = os.path.relpath(target, DOCUMENT_DIR).replace(chr(92), '/')
            links.append('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>')
        display(HTML(' · '.join(links)))

show_expression('expression_heatmaps')
profile = pd.read_csv(SUPPLEMENT / 'tables/marker_profiles.csv')
display(profile[['model_id', 'persona', 'marker_id', 'raw_count', 'linguistic_available',
                 'total', 'abstain_text', 'prevalence_pct', 'mean_response_density']].head(20))'''),
             nbformat.v4.new_markdown_cell('''## 5C. Kartu respons: istilah agama dalam konteks

Satu kartu per model memasangkan N, E, dan satu implicit pada **subset dan question_id yang sama**. Identitas dialokasikan tetap agar Islam, Christian, Jewish dan Atheist terwakili. Contoh dipilih secara purposif untuk memperlihatkan ragam marker E/implicit, **bukan sampel representatif**, bukan perbandingan peringkat antarmodel, dan tidak dipilih dari GT/akurasi. Daftar kandidat serta peringkat pemilihan tersedia pada CSV.

Sorotan memakai occurrence v3 dan offset karakter pada teks final asli. Warna hanya mengelompokkan pola kamus; fungsi, speaker dan referen yang belum pasti tetap belum terverifikasi. Abu-abu menunjukkan overlap kategori. Istilah yang belum ada di kamus tidak ditambahkan diam-diam. **Tidak disorot bukan berarti bukan bahasa agama.**

Gambar menampilkan cuplikan berkonteks sampai 100 kata dengan batas karakter eksplisit. Teks lengkap dan provenance setiap sorotan tersedia dalam panel di bawah serta HTML mandiri. Spasi gambar dirapikan, tetapi HTML/JSON mempertahankan teks asli. Label-only atau teks kosong pada pasangan tidak diganti dengan reasoning.'''),
             nbformat.v4.new_code_cell('''show_expression('expression_cards')
links = []
for filename, label in [('response_cards.html', 'Respons utuh berwarna (HTML)'),
                        ('response_cards.json', 'Teks dan provenance (JSON)'),
                        ('tables/card_selection_candidates.csv', 'Kandidat dan alasan pemilihan'),
                        ('tables/card_marker_counts.csv', 'Count dan density per respons/marker'),
                        ('tables/card_occurrences.csv', 'Seluruh occurrence contoh')]:
    relative = os.path.relpath(SUPPLEMENT / filename, DOCUMENT_DIR).replace(chr(92), '/')
    links.append('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>')
display(HTML(' · '.join(links)))
display(HTML((SUPPLEMENT / 'response_cards.html').read_text(encoding='utf-8')))''')]
    position = next((i for i, c in enumerate(result.cells)
                     if c.cell_type == 'markdown' and c.source.startswith('## 6.')), len(result.cells))
    result.cells[position:position] = cells
    result.metadata['expression_supplement'] = run_id
    return result


def load_cards(selected):
    wanted = set(selected.response_key)
    found, offset = [], 0
    columns = ['model', 'persona_folder', 'subset', 'question_id', 'response_content_full']
    for chunk in pd.read_csv(RAW_SOURCE, usecols=columns, dtype=str, keep_default_na=False, chunksize=10000):
        chunk = chunk.rename(columns={'model': 'model_id', 'persona_folder': 'persona', 'response_content_full': 'final_output'})
        chunk['source_row'] = np.arange(offset, offset + len(chunk))
        chunk['response_key'] = chunk[KEYS].apply(lambda row: json.dumps(row.tolist(), separators=(',', ':')), axis=1)
        found.append(chunk.loc[chunk.response_key.isin(wanted), ['response_key', 'final_output', 'source_row']])
        offset += len(chunk)
        print(f'  Teks sumber: {offset:,} baris diperiksa', flush=True)
    text = pd.concat(found, ignore_index=True)
    if set(text.response_key) != wanted or text.response_key.duplicated().any():
        raise ValueError('Selected raw final responses are missing or duplicated')
    cards = selected.merge(text, on='response_key', validate='one_to_one').to_dict('records')
    hits = []
    for chunk in pd.read_csv(SOURCE / 'occurrences/hits.csv.gz', chunksize=200000, keep_default_na=False):
        hits.append(chunk.loc[chunk.response_key.isin(wanted) & chunk.marker_id.isin(RELIGION_MARKERS)])
    occurrences = pd.concat(hits, ignore_index=True)
    for card in cards:
        if hashlib.sha256(card['final_output'].encode('utf-8')).hexdigest() != card['raw_final_sha256']:
            raise ValueError('Raw final text hash differs from frozen marker source')
        if re.search(r'</?think\b|<\|(?:start|channel|im_start|im_sep|message)\|>', card['final_output'], re.I):
            raise ValueError('Selected card contains technical/hidden channel delimiters; do not display as final-only')
        matching = occurrences[occurrences.response_key.eq(card['response_key'])]
        if len(matching) and not matching.raw_final_sha256.eq(card['raw_final_sha256']).all():
            raise ValueError('Occurrence text hash differs from response hash')
        card['hits'] = matching.to_dict('records')
        card['source_file'] = str(RAW_SOURCE.relative_to(ROOT)).replace('\\', '/')
        highlight_segments(card['final_output'], card['hits'])
        for marker in RELIGION_MARKERS:
            recorded = card[marker + '__count']
            observed = int(matching.marker_id.eq(marker).sum())
            if (pd.isna(recorded) and observed != 0) or (pd.notna(recorded) and observed != int(recorded)):
                raise ValueError('Occurrence count differs from response marker count')
    return cards, occurrences


def notebook_links(notebook, directory):
    checked = 0
    for cell in notebook.cells:
        blobs = [cell.source] if cell.cell_type == 'markdown' else []
        for output in cell.get('outputs', []):
            blobs.extend(value for mime, value in output.get('data', {}).items() if mime in ['text/html', 'text/markdown'])
        for blob in blobs:
            links = re.findall(r'href="([^"]+)"', blob) + re.findall(r'\]\(([^)]+)\)', blob)
            for link in links:
                link = html.unescape(link.strip('<>'))
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:|^#', link):
                    continue
                if not (Path(directory) / unquote(link.split('#')[0])).exists():
                    raise ValueError(f'Broken notebook link from {directory}: {link}')
                checked += 1
    return checked


def prepare_notebook(run_id, base_notebook=NOTEBOOK, active_directory=NOTEBOOK.parent, root=ROOT):
    base_notebook = Path(base_notebook).resolve()
    original = nbformat.read(base_notebook, as_version=4)
    if base_notebook.parent != Path(active_directory).resolve():
        original = rebase_notebook(original, base_notebook.parent, active_directory, root)
    return extend_notebook(original, run_id)


def build(run_id, output, base_notebook=NOTEBOOK):
    if output.exists():
        raise FileExistsError(f'Run already exists; use a new run ID: {output}')
    extended = prepare_notebook(run_id, base_notebook)
    old_hash = sha256(NOTEBOOK)
    inputs = [SOURCE / name for name in ['manifest.json', 'source_manifest.json', 'registry/markers.json',
              'statistics/overall_descriptive.csv', 'response_level/markers.csv.gz', 'occurrences/hits.csv.gz']]
    inputs += [RAW_SOURCE]
    if Path(base_notebook).resolve() != NOTEBOOK.resolve():
        inputs.append(Path(base_notebook).resolve())
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    print('[1/5] Menyiapkan profil lengkap dari tabel deskriptif beku', flush=True)
    table = heatmap_table(pd.read_csv(SOURCE / 'statistics/overall_descriptive.csv'))
    registry = json.loads((SOURCE / 'registry/markers.json').read_text(encoding='utf-8'))
    if set(m['marker_id'] for m in registry if m['marker_id'].startswith('religion_')) != set(RELIGION_MARKERS):
        raise ValueError('Religion marker list does not match the frozen registry')
    columns = KEYS + ['response_key', 'raw_final_sha256', 'answer_status', 'surface_status', 'word_count']
    columns += [marker + suffix for marker in RELIGION_MARKERS for suffix in ['__count', '__density']]
    rows = pd.read_csv(SOURCE / 'response_level/markers.csv.gz', usecols=columns)
    print('[2/5] Memilih pasangan contoh dan memeriksa hash/offset teks', flush=True)
    selected, candidates = select_card_rows(rows)
    cards, occurrences = load_cards(selected)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tables').mkdir()
    table.to_csv(output / 'tables/marker_profiles.csv', index=False)
    candidates.to_csv(output / 'tables/card_selection_candidates.csv', index=False)
    occurrences.to_csv(output / 'tables/card_occurrences.csv', index=False)
    counts = []
    for card in cards:
        for marker in RELIGION_MARKERS:
            counts.append(dict(response_key=card['response_key'], card_id=card['card_id'], model_id=card['model_id'],
                               persona=card['persona'], subset=card['subset'], question_id=card['question_id'],
                               answer_status=card['answer_status'], surface_status=card['surface_status'],
                               word_count=card['word_count'], marker_id=marker,
                               raw_count=card[marker + '__count'], density_per_100_words=card[marker + '__density']))
    pd.DataFrame(counts).to_csv(output / 'tables/card_marker_counts.csv', index=False)
    # JSON uses null for unavailable density, never the nonstandard NaN literal.
    cards = json.loads(pd.Series(cards).to_json(orient='values', force_ascii=False))
    write_json(output / 'response_cards.json', cards)
    print('[3/5] Membuat heatmap, kartu PNG/SVG, dan HTML respons utuh', flush=True)
    index = render_expression(output, table, cards)
    (output / 'code_snapshot').mkdir()
    for filename in ['build_religion_ethics_expression.py', 'religion_ethics_expression.py', 'religion_ethics_expression_plots.py',
                     'religion_ethics_gsm8k_style_data.py', 'religion_ethics_gsm8k_style_notebook.py', 'build_religion_ethics_gsm8k_style.py']:
        shutil.copy2(Path(__file__).with_name(filename), output / 'code_snapshot' / filename)
    shutil.copy2(Path(__file__).with_name('tests') / 'test_religion_ethics_expression.py',
                 output / 'code_snapshot/test_religion_ethics_expression.py')
    artifacts = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'manifest.json', dict(run_id=run_id, source_bundle=SOURCE.relative_to(ROOT).as_posix(),
               source_hashes=hashes, artifact_hashes=artifacts, previous_active_notebook_sha256=old_hash,
               status='candidate_pending_independent_human_validation', source_responses=len(rows),
               religious_markers=len(RELIGION_MARKERS), unique_profile_cells=len(table), figures=len(index),
               card_response_count=len(cards), card_identities=CARD_IDENTITIES,
               selection='purposive_max_distinct_markers_then_raw_hits_then_shorter_combined_words_then_subset_question',
               hidden_reasoning_used=False, detector_changed=False, scoring_changed=False, inference_recomputed=False))
    print('[4/5] Mengeksekusi notebook lama beserta dua bagian tambahan', flush=True)
    executed = execute_notebook(extended)
    report = rebase_notebook(executed, NOTEBOOK.parent, output, ROOT)
    for filename, notebook in [('religion_ethics_expression.workspace.ipynb', executed), ('religion_ethics_expression.ipynb', report)]:
        with (output / filename).open('x', encoding='utf-8') as handle:
            nbformat.write(notebook, handle)
    print('[5/5] Verifikasi seluruh angka, gambar, navigasi, dan sumber', flush=True)
    frozen_summary = pd.read_csv(SOURCE / 'statistics/overall_descriptive.csv').set_index(['model_id', 'persona', 'marker_id'])
    for record in index.itertuples():
        with Image.open(output / record.png) as image:
            image.verify()
        if record.section == 'expression_heatmaps':
            saved = pd.read_csv(output / record.table)
            expected = frozen_summary.loc[
                pd.MultiIndex.from_frame(saved[['model_id', 'persona', 'marker_id']])]
            for column in ['raw_count', 'prevalence_pct', 'mean_response_density', 'linguistic_available', 'total', 'abstain_text']:
                source_column = 'prevalence_text' if column == 'prevalence_pct' else column
                np.testing.assert_allclose(saved[column], expected[source_column], rtol=1e-12, atol=1e-12, equal_nan=True)
    for relative, expected in hashes.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Input changed: ' + relative)
    verification = dict(figures=len(index), profile_cells=len(table), plotted_heatmap_cells=int(index.loc[index.section.eq('expression_heatmaps'), 'rows'].sum()),
                        card_responses=len(cards), validated_highlight_spans=len(occurrences), source_files_unchanged=len(hashes),
                        notebook_code_cells_executed=sum(c.cell_type == 'code' for c in executed.cells),
                        embedded_png=sum('image/png' in out.get('data', {}) for c in executed.cells for out in c.get('outputs', [])),
                        active_links=notebook_links(executed, NOTEBOOK.parent), report_links=notebook_links(report, output),
                        workspace_notebook_sha256=sha256(output / 'religion_ethics_expression.workspace.ipynb'))
    write_json(output / 'verification.json', verification)
    print(json.dumps(verification), flush=True)


def install(run_id, output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    if sha256(NOTEBOOK) != manifest['previous_active_notebook_sha256']:
        raise ValueError('Active notebook changed since build; refusing overwrite')
    built = output / 'religion_ethics_expression.workspace.ipynb'
    if sha256(built) != verification['workspace_notebook_sha256']:
        raise ValueError('Built notebook changed')
    for relative, expected in manifest['artifact_hashes'].items():
        if sha256(output / relative) != expected:
            raise ValueError('Artifact changed: ' + relative)
    target = NOTEBOOK.parent / 'figures' / f'expression_{run_id}'
    archive = NOTEBOOK.parent / 'archive' / f'before_expression_{run_id}'
    if target.exists() or archive.exists() or (output / 'installation_verification.json').exists():
        raise FileExistsError('Installation destination already exists')
    shutil.copytree(output / 'figures', target)
    archive.mkdir(parents=True, exist_ok=False)
    old = rebase_notebook(nbformat.read(NOTEBOOK, as_version=4), NOTEBOOK.parent, archive, ROOT)
    with (archive / NOTEBOOK.name).open('x', encoding='utf-8') as handle:
        nbformat.write(old, handle)
    write_json(archive / 'manifest.json', dict(original_sha256=sha256(NOTEBOOK), only_navigation_paths_adjusted=True))
    shutil.copy2(built, NOTEBOOK)
    for path in target.iterdir():
        if sha256(path) != sha256(output / 'figures' / path.name):
            raise ValueError('Installed figure differs: ' + str(path))
    result = dict(active_notebook=str(NOTEBOOK), archive=str(archive / NOTEBOOK.name), figures=str(target),
                  png_count=len(list(target.glob('*.png'))), svg_count=len(list(target.glob('*.svg'))),
                  active_links=notebook_links(nbformat.read(NOTEBOOK, as_version=4), NOTEBOOK.parent),
                  archive_links=notebook_links(old, archive), all_figure_hashes_match=True)
    write_json(output / 'installation_verification.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--base-notebook', type=Path, default=NOTEBOOK,
                        help='Explicit notebook base for reproduction, e.g. the archived pre-supplement notebook.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run-id')
    output = ROOT / 'result/ETHICS/reports' / f'religion_ethics_expression_{args.run_id}'
    if args.install:
        install(args.run_id, output)
    else:
        build(args.run_id, output, args.base_notebook)


if __name__ == '__main__':
    main()
