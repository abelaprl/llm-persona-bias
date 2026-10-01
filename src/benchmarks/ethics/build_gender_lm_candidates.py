"""Build two additive, executed notebooks from frozen linguistic-marker bundles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pandas as pd

from .gender_lm_reference_data import ROOT, ACTIVE, digest

DEFAULT_OUTPUT = ROOT / 'notebooks/gender/linguistic marker/candidate'
NOTEBOOKS = {name: f'{name}_Gender_Linguistic_Markers_candidate.ipynb' for name in ['GSM8K', 'ETHICS']}


def make_notebook(benchmark, figures):
    """Make a reader-first notebook whose embedded images survive sharing the file."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                             language_info=dict(name='python'), candidate_figure_ids=[r['id'] for r in figures])
    intro = (
        f'# Gender {benchmark} — candidate presentasi linguistic marker\n\n'
        'Alur: **cakupan → panjang respons → profil absolut → perbedaan persona '
        '→ bentuk bahasa nyata → detail**. Warna mewakili enam model yang sama; '
        'E/I1/I2/I3 adalah kondisi persona, bukan urutan waktu.\n\n'
        'Ini revisi penyajian, bukan ekstraksi baru atau validasi kamus. Angka sumber, '
        'aturan, pairing, dan koreksi statistik tidak diubah. Semua hasil tetap **candidate**; '
        'tidak ada pemilihan respons berdasarkan benar/salah sebagai analisis utama. '
        'Tidak memakai hidden reasoning.\n\n'
        + ('GSM8K memakai rank-biserial untuk ringkasan efek numerik. Nilai ekstrem pada sedikit '
           'pasangan berubah tidak membuktikan perubahan luas. CI median-delta sumber '
           '**tidak** dipasang pada sumbu rank.\n\n' if benchmark == 'GSM8K' else
           'ETHICS mempertahankan selisih rerata density atau prevalence dalam poin persentase. '
           'CI adalah 95% marginal, bukan CI terkoreksi Holm; adjusted_p sumber tersedia di CSV. '
           'Overall di sini adalah marker, **bukan akurasi gabungan ETHICS**. '
           'Efek memakai pasangan berpenjelasan tersedia; perbedaan coverage perlu diperhatikan.\n\n')
        + '[Galeri kedua benchmark](index.html) · [Catatan dan cara membangun ulang](README.md)\n\n'
        'Gambar selebar 16 cm, DejaVu Sans minimum 9 pt; gunakan PNG/SVG pada ukuran tersebut. '
        'CSV di samping setiap gambar menyimpan angka dan provenance.'
    )
    cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
        "options = [Path.cwd(), Path.cwd() / 'notebooks/gender/linguistic marker/candidate']\n"
        "CANDIDATE = next(p for p in options if (p / 'figure_index.csv').is_file())\n"
        "figure_index = pd.read_csv(CANDIDATE / 'figure_index.csv')\n"
        f"display(figure_index.loc[figure_index.benchmark.eq({benchmark!r}), "
        "['id', 'title', 'section', 'png', 'csv']])"
    )]
    previous_section = None
    for record in figures:
        if record['section'] != previous_section:
            heading = 'Cerita utama' if record['section'] == 'main' else 'Lampiran: rincian lengkap'
            cells.append(nbformat.v4.new_markdown_cell('## ' + heading))
            previous_section = record['section']
        description = (f"### {record['title']}\n\n{record['question']}\n\n{record['caption']}\n\n"
                       f"[PNG]({record['png']}) · [SVG]({record['svg']}) · [CSV]({record['csv']})")
        if record.get('note'):
            description += '\n\n' + record['note']
        cells.append(nbformat.v4.new_markdown_cell(description))
        cells.append(nbformat.v4.new_code_cell(
            f"display(Image(filename=str(CANDIDATE / {record['png']!r}), width=760))"))
        if record.get('support_csv'):
            cells.append(nbformat.v4.new_markdown_cell(
                'Dukungan per titik tersedia lengkap pada '
                f"[tabel pasangan]({record['support_csv']}). Contoh ekstrem dengan ≤5 pasangan berubah "
                'ditampilkan di bawah; ini diagnostik kelangkaan, bukan seleksi untuk pengujian.'))
            cells.append(nbformat.v4.new_code_cell(
                f"support = pd.read_csv(CANDIDATE / {record['support_csv']!r})\n"
                "diagnostic = support.loc[support.n_nonzero.le(5) & support.rank_biserial.abs().ge(.9)]\n"
                "columns = ['model_id', 'panel_scenario', 'outcome_id', 'n_nonzero', 'n_pairs', 'rank_biserial', 'p_holm']\n"
                "display(diagnostic[columns].reset_index(drop=True))"))
        if record.get('supplemental_tables'):
            cells.append(nbformat.v4.new_markdown_cell(
                '<details><summary>Tabel pendamping dan sumber lengkap</summary>\n\n' +
                '\n'.join(f"- [{item['title']}]({item['csv']})" for item in record['supplemental_tables']) +
                '\n\n</details>'))
    if benchmark == 'ETHICS':
        cells.append(nbformat.v4.new_markdown_cell(
            '## Cakupan lengkap dan konteks\n\n'
            '- [Indeks semua 41 marker](tables/ethics_visual_coverage.csv)\n'
            '- [Profil semua marker: 6 model × 9 persona, raw count + denominator](tables/ethics_all_marker_profiles.csv)\n'
            '- [Semua 984 kontras direct](tables/ethics_all_direct_effects.csv)\n'
            '- [Kontras terhadap netral dan implisit–eksplisit](tables/ethics_other_contrasts.csv)\n'
            '- [Kamus dan sumber teoritis; adaptasi buatan sendiri ditandai](tables/ethics_marker_catalog.csv)\n'
            '- [Explorer sumber: profil/perbandingan seluruh kelompok](ethics_explorer.html)\n'
            '- [Dua triplet contoh beserta teks utuh](ethics_examples.html)\n'
            '- [Bentuk ungkapan enam model](tables/ethics_expression_forms.csv)\n\n'
            'Marker adalah proxy leksikal yang belum selesai diaudit; jumlah kategori bukan '
            'ukuran kekayaan semantik menyeluruh. Semua respons diinventarisasi, termasuk abstain; '
            'respons tanpa penjelasan tidak diimputasi menjadi nol. Rincian per subset tetap '
            'tersedia pada CSV sumber; ringkasan overall mempertahankan pembobotan sumber.'))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Batas perbandingan kedua benchmark\n\n'
        'Gaya visual diseragamkan, tetapi instrumen dan populasi tetap berbeda. '
        '16 indikator utama GSM8K bukan padanan 41 marker kamus ETHICS. '
        'Rank-biserial tidak dapat dibandingkan langsung dengan selisih density/prevalence. '
        'Tidak ada model yang dipool sebagai efek umum, tidak ada uji baru atas token yang '
        'dipilih untuk ilustrasi, dan tidak ada klaim instrumen sudah tervalidasi.'))
    notebook.cells = cells
    return notebook


def protected_paths(inputs):
    """Snapshot actual inputs and old active artifacts, excluding this new candidate."""
    paths = set(inputs)
    for directory in [ACTIVE, ROOT / 'notebooks/gender/linguistic marker/figures']:
        paths.update(p for p in directory.rglob('*') if p.is_file())
    paths.add(ROOT / 'notebooks/gender/linguistic marker/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb')
    return sorted(paths)


def write_gallery(output, figures):
    chunks = ['<!doctype html><html lang="id"><meta charset="utf-8">'
              '<title>Candidate Gender — GSM8K dan ETHICS</title>'
              '<style>body{font:16px system-ui;line-height:1.55;max-width:980px;margin:32px auto;padding:0 20px}'
              'img{display:block;max-width:100%;width:650px;margin:18px auto}section{border-top:1px solid #ccd;padding:20px 0}'
              'summary{cursor:pointer;font-weight:bold}a{color:#245ac0}.note{background:#f3f6fa;padding:15px}</style>'
              '<h1>Gender: dua candidate presentasi</h1><p class="note">'
              'GSM8K dan ETHICS memiliki alur visual seragam, tetapi metriknya tidak disamakan. '
              'Semua hasil tetap candidate. File lama tidak diganti. '
              '<a href="README.md">Catatan metodologi dan reproduksi</a>.</p>']
    for benchmark, notebook in NOTEBOOKS.items():
        chunks.append(f'<h2 id="{benchmark}">{benchmark}</h2><p><a href="{notebook}">Notebook {benchmark}</a></p>')
        for row in [r for r in figures if r['benchmark'] == benchmark]:
            body = (f'<section id="{html.escape(row["id"])}"><h3>{html.escape(row["title"])}</h3>'
                    f'<p>{html.escape(row["question"])}</p><p>{html.escape(row["caption"])}</p>'
                    f'<img loading="lazy" src="{html.escape(row["png"])}" alt="{html.escape(row["title"])}">'
                    f'<p><a href="{row["png"]}">PNG</a> · <a href="{row["svg"]}">SVG</a> · '
                    f'<a href="{row["csv"]}">CSV</a></p>')
            if row.get('note'):
                body += '<p class="note">' + html.escape(row['note']) + '</p>'
            if row.get('support_csv'):
                body += f'<p><a href="{row["support_csv"]}">Jumlah pasangan dan Holm per titik</a></p>'
            body += '</section>'
            chunks.append(body if row['section'] == 'main' else
                          '<details><summary>' + html.escape(row['title']) + ' — detail</summary>' + body + '</details>')
    chunks.append('<p><a href="ethics_examples.html">Contoh ETHICS utuh</a> · '
                  '<a href="ethics_explorer.html">Explorer ETHICS lengkap</a></p></html>')
    (output / 'index.html').write_text(''.join(chunks), encoding='utf-8')


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output sudah ada; gunakan path baru: {output}')
    from . import gender_lm_candidate_ethics as ethics
    from . import gender_lm_candidate_gsm8k as gsm8k
    inputs = sorted(set(ethics.input_paths() + gsm8k.input_paths()))
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    protected = protected_paths(inputs)
    before = {str(path): digest(path) for path in protected}
    output.mkdir(parents=True, exist_ok=False)
    for directory in ['figures', 'tables']:
        (output / directory).mkdir()
    manifest = dict(status='building', built_at=datetime.now(timezone.utc).isoformat(),
                    output=str(output), source_data_modified=False, extraction_changed=False,
                    inference_recomputed=False, input_paths=[str(p) for p in inputs])
    def persist_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    persist_manifest()
    try:
        figures = []
        for benchmark, module in [('GSM8K', gsm8k), ('ETHICS', ethics)]:
            print(f'Rendering {benchmark}...', flush=True)
            records = module.build(output)
            figures.extend(dict(record, benchmark=benchmark) for record in records)
        pd.DataFrame(figures).drop(columns=['geometry']).to_csv(output / 'figure_index.csv', index=False)
        (output / 'figure_manifest.json').write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding='utf-8')
        examples = (ethics.PACKAGE / 'examples.html').read_text(encoding='utf-8')
        for original, replacement in [('explorer.html', 'ethics_explorer.html'),
                                      ('expression_forms.csv', 'tables/ethics_expression_forms.csv'),
                                      ('example_spans.csv', 'tables/ethics_example_spans.csv')]:
            examples = examples.replace(f'href="{original}', f'href="{replacement}')
        (output / 'ethics_examples.html').write_text(examples, encoding='utf-8')
        explorer = (ethics.PACKAGE / 'explorer.html').read_text(encoding='utf-8')
        explorer = explorer.replace('href="examples.html', 'href="ethics_examples.html')
        explorer = explorer.replace('href="visual_coverage.csv', 'href="tables/ethics_source_visual_coverage.csv')
        (output / 'ethics_explorer.html').write_text(explorer, encoding='utf-8')
        (output / 'tables/ethics_source_visual_coverage.csv').write_bytes(
            (ethics.PACKAGE / 'visual_coverage.csv').read_bytes())
        write_gallery(output, figures)
        readme = '# Candidate Gender GSM8K dan ETHICS\n\n'
        readme += '[Buka galeri](index.html). Dua notebook terpisah, satu folder figures utama.\n\n'
        for benchmark, filename in NOTEBOOKS.items():
            selected = [r for r in figures if r['benchmark'] == benchmark]
            readme += (f'- [{benchmark}]({filename}): {sum(r["section"] == "main" for r in selected)} '
                       f'gambar utama + {sum(r["section"] == "detail" for r in selected)} detail.\n')
        readme += ('\n## Keputusan visual dan batas metodologi\n\n'
                   '- Cerita utama dipisahkan dari lampiran lengkap, bukan menghapus marker.\n'
                   '- Model memakai warna tetap dan posisi titik terpisah. Tidak ada berlian median lintas-model.\n'
                   '- Lebar cetak 16 cm; DejaVu Sans minimum 9 pt; PNG 240 dpi dan SVG tersedia.\n'
                   '- GSM8K: semua 16 indikator numerik utama ditampilkan; dukungan pasangan tercatat per titik. '
                   'Titik kosong menandai <20 pasangan berubah, bukan ketidaksignifikanan. '
                   'Rank-biserial tidak diberi CI median-delta.\n'
                   '- ETHICS: 11 indikator ringkasan dipilih di muka, lalu semua 41 marker memiliki gambar detail. '
                   'Prevalence profil tetap persen respons; efek density per100kata dan prevalence pp dipisah. '
                   'Zoom B08 hanya I1–I3 dan dinyatakan jelas; sumber nilainya identik.\n'
                   '- CI ETHICS 95% marginal, bukan CI terkoreksi Holm; keputusan inferensial tetap adjusted_p sumber.\n'
                   '- Profil mencakup semua penjelasan tersedia, termasuk abstain. Missing/technical-only tidak menjadi nol. '
                   'ETHICS verdict-only tidak masuk denominator marker, tetapi 0 kata pada tabel panjang respons sumber.\n'
                   '- Raw count dan denominator disimpan pada tabel profil/marker, bukan hanya density. '
                   'Sumber kamus dan adaptasi sendiri tetap dicantumkan di tabel katalog.\n'
                   '- Overall ETHICS bukan akurasi gabungan. Komposisi subset/pasangan dan bobot sumber dipertahankan; '
                   'tabel per subset tetap ada di sumber.\n'
                   '- Token dan contoh kalimat adalah ilustrasi eksploratif/purposif. Referen gender belum selesai diaudit. '
                   'Kedua instrumen masih candidate; visualisasi bukan bukti validitas.\n\n'
                   '## Sumber beku\n\n')
        for source in [gsm8k.BUNDLE, gsm8k.SOURCE, ethics.SOURCE, ethics.PACKAGE]:
            link = Path(os.path.relpath(source, output)).as_posix().replace(' ', '%20')
            readme += f'- [{source.name}]({link})\n'
        readme += ('\n## Reproduksi\n\nDari `D:\\TugasAkhir`, jalankan dengan **path output baru**:\n\n'
                   '```powershell\npython -m src.benchmarks.ethics.build_gender_lm_candidates '
                   '--output "notebooks/gender/linguistic marker/candidate_rerun"\n```\n\n'
                   'Builder menolak folder yang sudah ada. Notebook membaca tabel dan menanamkan PNG; '
                   'builder di atas merender ulang semuanya dari sumber beku dan mengeksekusi kedua notebook. '
                   'Tidak ada API/model call atau perubahan aturan ekstraksi.\n\n'
                   '`source_hashes.json` membandingkan sumber dan artefak lama sebelum/sesudah; '
                   '`code_hashes.json` mengunci kode pembuat; `verification.json` mencatat jumlah gambar dan hasil eksekusi.\n')
        (output / 'README.md').write_text(readme, encoding='utf-8')
        notebook_checks = {}
        for benchmark, filename in NOTEBOOKS.items():
            print(f'Executing {benchmark} notebook...', flush=True)
            selected = [record for record in figures if record['benchmark'] == benchmark]
            notebook = make_notebook(benchmark, selected)
            NotebookClient(notebook, timeout=180, kernel_name='python3',
                           resources={'metadata': {'path': str(output)}}).execute()
            errors = [out for cell in notebook.cells if cell.cell_type == 'code'
                      for out in cell.outputs if out.output_type == 'error']
            png_count = sum('image/png' in out.get('data', {}) for cell in notebook.cells
                            if cell.cell_type == 'code' for out in cell.outputs)
            if errors or png_count != len(selected):
                raise ValueError(f'Notebook incomplete: {benchmark}')
            nbformat.write(notebook, output / filename)
            notebook_checks[benchmark] = dict(errors=len(errors), embedded_pngs=png_count)
        after = {str(path): digest(path) for path in protected}
        if after != before:
            raise ValueError('A protected source or old artifact changed')
        (output / 'source_hashes.json').write_text(json.dumps(dict(before=before, after=after, unchanged=True),
                                                            indent=2), encoding='utf-8')
        code = [Path(__file__), Path(ethics.__file__), Path(gsm8k.__file__)]
        code += [Path(__file__).with_name(name) for name in ['gender_lm_candidate_common.py',
                 'gender_lm_reference_data.py', 'gender_lm_reference_plots.py',
                 'gender_lm_presentation_v2_data.py', 'gender_lm_presentation_v2_plots.py',
                 'restyle_gender_lm_gsm8k.py']]
        (output / 'code_hashes.json').write_text(json.dumps({str(p): digest(p) for p in code}, indent=2), encoding='utf-8')
        verification = dict(notebooks=notebook_checks, protected_files=len(protected),
                            old_files_unchanged=True, figures=len(figures),
                            min_font_pt=min(r['geometry']['min_font_pt'] for r in figures),
                            widths_cm=sorted({r['geometry']['width_cm'] for r in figures}),
                            geometry_passed=True, source_status='candidate')
        (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        manifest.update(status='complete_candidate_presentation', figures=len(figures), notebooks=NOTEBOOKS)
        persist_manifest()
        return verification
    except Exception as error:
        manifest.update(status='incomplete_do_not_use', error=str(error))
        persist_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output), indent=2))


if __name__ == '__main__':
    main()
