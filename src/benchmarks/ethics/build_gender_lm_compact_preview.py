"""Build a new, non-destructive 14-figure preview from frozen gender evidence."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.text import Text
import nbformat
import numpy as np
import pandas as pd

from src.benchmarks.ethics import gender_lm_compact_plots as plots
from src.benchmarks.ethics.gender_lm_compact_data import (
    atlas_data, subset_view, normalization_view, interaction_view,
)
from src.benchmarks.ethics.gender_lm_diagnostic_data import (
    direct_rows, support_summary, PERSONAS, CONDITIONS, INDICATORS,
)

SOURCE = ROOT / 'result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3'
DIAGNOSTICS = ROOT / 'result/ETHICS/reports/gender_lm_diagnostics_20260919_r1'
ACTIVE = ROOT / 'notebooks/gender/linguistic marker/ETHICS'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require_new_output(path):
    if Path(path).exists():
        raise FileExistsError(f'Preview requires a new output directory: {path}')


def relative(path, output):
    return Path(os.path.relpath(path, output)).as_posix()


class Writer:
    def __init__(self, output):
        require_new_output(output)
        self.output = output
        (output / 'figures').mkdir(parents=True)
        (output / 'tables').mkdir()
        self.index = []

    def add(self, figure, title, caption, table):
        number = len(self.index) + 1
        name = f'{number:02d}'
        figure.canvas.draw()
        texts = [t for t in figure.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 8.5
        assert np.isclose(figure.get_figwidth() * 2.54, 16)
        assert figure.get_figheight() * 2.54 <= 26.01
        png = self.output / 'figures' / f'{name}.png'
        svg = png.with_suffix('.svg')
        figure.savefig(png, dpi=300)
        figure.savefig(svg)
        csv = self.output / 'tables' / f'{name}.csv'
        table.to_csv(csv, index=False)
        # Round-trip evidence values exactly; CSV dtype inference is not an estimand.
        reloaded = pd.read_csv(csv, float_precision='round_trip')
        assert len(reloaded) == len(table)
        for col in table.select_dtypes(include=['floating']).columns:
            np.testing.assert_allclose(table[col], reloaded[col], rtol=0, atol=0, equal_nan=True)
        self.index.append(dict(figure_id=name, title=title, caption=caption,
                               png_path=f'figures/{name}.png', svg_path=f'figures/{name}.svg',
                               source_csv=f'tables/{name}.csv', source_rows=len(table),
                               png_sha256=digest(png), svg_sha256=digest(svg),
                               source_sha256=digest(csv), width_cm=16,
                               height_cm=figure.get_figheight() * 2.54, dpi=300,
                               min_font_pt=min(t.get_fontsize() for t in texts)))
        plt.close(figure)
        print(f'{name}/14 {title}', flush=True)


def interaction_figure(data, registry):
    plots.style()
    fig = plt.figure(figsize=(16 / 2.54, 26 / 2.54))
    configurations = [('density', [.28, .45, .705, .46], [.31, .395, .61, .011],
                       'Densitas: selisih perubahan / 100 kata'),
                      ('presence', [.28, .15, .705, .16], [.31, .095, .61, .011],
                       'Prevalensi: selisih perubahan (poin persentase)')]
    columns = [(m, c) for m in plots.MODELS for c in ['1−2', '1−3', '2−3']]
    for measurement, position, bar_position, title in configurations:
        rows = registry.loc[registry.measurement.eq('binary' if measurement == 'presence' else measurement), 'marker_id'].tolist()
        part = data.loc[data.measurement.eq(measurement)].copy()
        part['cell'] = list(zip(part.model_id, part.condition))
        values = plots.matrix(part, rows, columns, 'cell', 'effect')
        p = plots.matrix(part, rows, columns, 'cell', 'adjusted_p').to_numpy(float)
        symbols = np.where(p < .05, '*', '').astype(object)
        symbols[~np.isfinite(values)] = 'NA'
        ax = fig.add_axes(position)
        limit = max(.001, float(np.nanmax(np.abs(values))))
        im = plots.heatmap(ax, values, [plots.SHORT[k] for k in rows], [c for _, c in columns],
                           limit=limit, diverging=True, symbols=symbols)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(title, loc='left', fontsize=9, pad=24)
        for i, model in enumerate(plots.MODELS):
            ax.text(i * 3 + 1, 1.012, plots.CODES[model], transform=ax.get_xaxis_transform(), ha='center')
            if i:
                ax.axvline(i * 3 - .5, color='#111827', lw=.7)
        fig.colorbar(im, cax=fig.add_axes(bar_position), orientation='horizontal')
    fig.suptitle('Perubahan selisih gender antar-skenario implisit', fontsize=12, y=.975)
    fig.text(.5, .035, '(Woman−Man) Iₐ − (Woman−Man) Iᵦ; * p Holm < 0,05.\nSkala warna terpisah menurut satuan; abu-abu = NA.', ha='center')
    return fig


def package(writer, registry, protected):
    output = writer.output
    appendix = []
    for index_name, bundle in [('figure_index.csv', ROOT / 'result/ETHICS/reports/gender_lm_gsm8k_style_20260918_final'),
                               ('diagnostic_figure_index.csv', DIAGNOSTICS)]:
        index = pd.read_csv(ACTIVE / index_name)
        for row in index.to_dict('records'):
            image_path = ACTIVE / row['png_path']
            table_path = ACTIVE / row['source_csv']
            if not table_path.exists():
                table_path = bundle / row['source_csv']
            assert image_path.is_file() and table_path.is_file()
            appendix.append(dict(title=row['title'], png_path=relative(image_path, output),
                                 source_csv=relative(table_path, output)))
    pd.DataFrame(writer.index).to_csv(output / 'figure_index.csv', index=False)
    pd.DataFrame(appendix).to_csv(output / 'appendix_index.csv', index=False)
    intro = (
        '# Pratinjau ringkas penanda linguistik Gender ETHICS\n\n'
        '14 figur utama · 41 penanda · 6 model · 9 persona · candidate, belum validasi manusia. '
        'Notebook aktif dan 168 grafik lama tidak diubah.\n\n'
        'Unit observasi: satu respons. Teks final tersedia, termasuk abstain; bukan reasoning channel. '
        'Profil memakai prevalensi (%) agar satuannya sama. Inferensi tetap memakai densitas '
        '(kejadian/100 kata) atau presence sesuai registry. Jumlah mentah tersedia dalam tabel profil. '
        'Respons missing/teknis bukan nol; verdict-only mempunyai penjelasan sepanjang nol kata.\n\n'
        'N = neutral; M/W = Man/Woman; E = eksplisit; I1–I3 = skenario implisit. '
        'L31/L33 = Llama 3.1/3.3; MIN = Ministral; DSR1 = DeepSeek R1; GPT = GPT-OSS; QWN = Qwen3 Next.\n\n'
        'Warna efek biru = negatif, merah muda = positif. Semua efek, termasuk nonsignifikan, '
        'ditampilkan; * berarti p Holm < 0,05, bukan CI yang dikoreksi. CI nominal, n_pairs, '
        'n_changed, dan p lengkap ada di tabel serta lampiran. CI untuk selisih rerata tidak '
        'identik dengan estimand uji Wilcoxon. Tidak ada penghitungan ulang uji atau pooling p-value.\n\n'
        'Gambar 10–13 hanya peta arah efek, bukan uji robustness baru. Efek kecil dekat nol '
        'dapat berpindah tanda; lihat estimasi dan CI sebelum menyimpulkan. '
        'Perbandingan neutral dan E–I serta forest plot lengkap tetap tersedia di lampiran.\n'
    )
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_markdown_cell(intro)]
    sections = []
    for item in writer.index:
        title, caption = item['title'], item['caption']
        sections.append(f'<figure><a href="{item["svg_path"]}"><img src="{item["png_path"]}" loading="lazy" alt="{html.escape(title)}"></a>'
                        f'<figcaption><strong>Gambar {item["figure_id"]}. {html.escape(title)}.</strong> {html.escape(caption)} '
                        f'<a href="{item["source_csv"]}">Tabel lengkap</a> · <a href="{item["svg_path"]}">SVG</a></figcaption></figure>')
        encoded = base64.b64encode((output / item['png_path']).read_bytes()).decode('ascii')
        cell = nbformat.v4.new_markdown_cell(
            f'![{title}](attachment:figure.png)\n\nGambar {item["figure_id"]}. **{title}.** {caption}\n\n'
            f'[Tabel lengkap]({item["source_csv"]}) · [SVG]({item["svg_path"]})',
            attachments={'figure.png': {'image/png': encoded}})
        notebook.cells.append(cell)
    links = ''.join(f'<li>{html.escape(x["title"])}: <a href="{html.escape(x["png_path"])}">grafik</a> · '
                    f'<a href="{html.escape(x["source_csv"])}">tabel</a></li>' for x in appendix)
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        '## Lampiran dan reproduksi\n\n[168 grafik detail](preview.html#lampiran) · '
        '[Indeks lampiran](appendix_index.csv) · [Seluruh perbandingan](tables/comparisons.csv) · '
        '[Registry dan sumber kamus](registry.json).\n\n'
        'Notebook ini adalah pratinjau statis dengan PNG tertanam, bukan notebook analisis baru. '
        'Perintah reproduksi tercantum di README.md.'))
    nbformat.validate(notebook)
    nbformat.write(notebook, output / 'Gender_ETHICS_ringkas_preview.ipynb')
    safe_intro = ''.join(f'<p>{html.escape(p)}</p>' for p in intro.split('\n\n'))
    page = ('<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS — pratinjau ringkas</title>'
            '<style>body{font:16px/1.55 system-ui;color:#172033;max-width:880px;margin:36px auto;padding:0 20px}'
            'figure{margin:48px 0;page-break-inside:avoid}img{display:block;width:16cm;max-width:100%;height:auto;margin:auto}'
            'figcaption{margin:16px auto;max-width:16cm}a{color:#1654a5}summary{cursor:pointer;font-weight:bold}'
            '@media print{body{margin:0}figure{break-before:page}details{display:none}}</style><body>'
            + safe_intro + ''.join(sections) + '<details id="lampiran"><summary>168 grafik detail lama (klik untuk membuka)</summary><ul>'
            + links + '</ul></details><p><a href="tables/comparisons.csv">Seluruh perbandingan asli</a> · '
            '<a href="registry.json">Registry dan sumber kamus</a> · <a href="README.md">Metode & reproduksi</a></p></body></html>')
    (output / 'preview.html').write_text(page, encoding='utf-8')
    command = 'python -X utf8 -B -m src.benchmarks.ethics.build_gender_lm_compact_preview --output result/ETHICS/reports/NAMA_BARU'
    report = intro + '\n## Reproduksi\n\nDari D:/TugasAkhir:\n\n```powershell\n' + command + '\n```\n\n'
    report += ('## Batas dan pemetaan\n\nSPEC: peringkasan presentasi, tanpa mengubah ekstraksi/statistik.\n\n'
               'DESIGN: 2 cakupan/dukungan, 3 profil, 3 efek gender, 1 konteks, 2 subset, 2 normalisasi, 1 interaksi. '
               'IMPLEMENTATION: seluruh keluaran di folder baru; hasil lama ditautkan, bukan dihapus.\n\n'
               'CHECKLIST/TESTING: unit test untuk tanda, NA, nol, satuan, dan penolakan overwrite; '
               'round-trip nilai CSV; 14 figur 16 cm/300 dpi; hash sumber dan notebook aktif sebelum/sesudah. '
               'Rincian mesin di verification.json. INTEGRATION: pratinjau saja, belum dipasang ke notebook aktif.\n\n'
               'Pemadatan mengorbankan tampilan CI pada halaman utama, bukan menyembunyikan CI dari hasil. '
               'Lihat tabel/forest plot sebelum menginterpretasikan efek. Kamus dan sumber teoritis '
               'tetap mengikuti registry.json; penanda adaptasi/buatan sendiri tetap berstatus candidate. '
               'Ringkasan overall di sini adalah penanda linguistik, bukan akurasi ETHICS gabungan.\n')
    (output / 'README.md').write_text(report, encoding='utf-8')
    assert len(writer.index) == 14 and len(appendix) == 168
    assert all(digest(Path(path)) == old for path, old in protected.items())
    verification = dict(main_figures=14, appendix_figures=168, marker_count=len(registry),
                        source_and_active_files_unchanged=True, protected_files=len(protected),
                        exact_csv_float_roundtrip=True, native_width_cm=16, png_dpi=300,
                        min_font_pt=8.5, static_notebook_attachments=14,
                        inference_recomputed=False, human_validation_completed=False)
    (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(dict(protected_inputs=protected,
        generated_files={str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()},
        generator=relative(Path(__file__), output)), indent=2), encoding='utf-8')


def build(output):
    require_new_output(output)
    print('Checking frozen inputs and preserving active-file hashes...', flush=True)
    names = ['registry.json', 'marker_summary.csv', 'comparisons.csv', 'interaction_comparisons.csv',
             'response_length_summary.csv', 'report_status_coverage.csv']
    reference_path = DIAGNOSTICS / 'diagnostic_tables/diagnostic_gender_reference_context.csv'
    protected_paths = [SOURCE / n for n in names] + [reference_path]
    protected_paths += list(ACTIVE.glob('*.ipynb')) + list((ACTIVE / 'figures').glob('*.png'))
    protected_paths += [Path(__file__), Path(__file__).with_name('gender_lm_compact_data.py'),
                        Path(__file__).with_name('gender_lm_compact_plots.py')]
    protected = {str(p): digest(p) for p in protected_paths}
    print('Reading frozen tables...', flush=True)
    read = lambda n: pd.read_csv(SOURCE / n, float_precision='round_trip')
    registry = pd.DataFrame(json.loads((SOURCE / 'registry.json').read_text(encoding='utf-8')))
    comparisons, interactions = read('comparisons.csv'), read('interaction_comparisons.csv')
    primary = direct_rows(comparisons, registry)
    assert len(primary) == 4920 and primary.marker_id.nunique() == 41
    assert set(primary.model_id) == set(plots.MODELS)
    writer = Writer(output)
    print('Rendering 14 native-size figures...', flush=True)
    # Preserve the complete frozen tables, including contrasts not shown on the main pages.
    for name in names:
        target = output / name if name == 'registry.json' else output / 'tables' / name
        target.write_bytes((SOURCE / name).read_bytes())
    status = read('report_status_coverage.csv').query("scope == 'overall'")
    lengths = read('response_length_summary.csv').query("scope == 'overall'")
    coverage = status.pivot(index=['model_id', 'persona'], columns='text_status', values='count').fillna(0)
    coverage['available_percent'] = 100 * (coverage.get('substantive_explanation', 0) + coverage.get('verdict_only', 0)) / coverage.sum(axis=1)
    coverage = coverage.reset_index()
    overview = coverage.merge(lengths, on=['model_id', 'persona'], validate='one_to_one')
    writer.add(plots.overview_figure(coverage, lengths, PERSONAS), 'Cakupan dan panjang respons',
               'Teks tersedia mencakup verdict-only. Median mencakup penjelasan nol kata pada verdict-only; missing/teknis dikecualikan. '
               'Persentase dibulatkan hanya pada gambar; jumlah tiap status dan kuantil panjang ada di tabel.', overview)
    support = support_summary(primary)
    writer.add(plots.support_figure(support, CONDITIONS), 'Dukungan pasangan',
               'Jumlah pasangan tidak dijumlahkan antar-marker. n_changed adalah pasangan yang berbeda nilai, '
               'bukan jumlah respons tersedia. Tabel memuat n_pairs dan denominator.', support)
    atlas = atlas_data(read('marker_summary.csv'))
    for scope, title in [('general_ethics', 'Penanda umum'), ('ethics_specific', 'Penanda etika'), ('gender_specific', 'Penanda gender')]:
        markers = registry.loc[registry.registry_scope.eq(scope), 'marker_id'].tolist()
        part = atlas.loc[atlas.marker_id.isin(markers)]
        fig = plots.model_panels(part, markers, PERSONAS, 'persona', 'value', title,
                                 'Prevalensi (% respons dengan ≥1 hit); abu-abu = NA.', column_labels=plots.PERSONA_CODES)
        writer.add(fig, title, 'Profil deskriptif prevalensi dengan skala 0–100% yang sama pada semua model. '
                   'Tabel juga menyimpan jumlah mentah, rerata jumlah, dan densitas per 100 kata; '
                   'ini bukan penggantian ukuran utama untuk uji statistik.', part)
    density = registry.loc[registry.measurement.eq('density'), 'marker_id'].tolist()
    binary = registry.loc[registry.measurement.eq('binary'), 'marker_id'].tolist()
    overall = primary.loc[primary.scope.eq('overall')]
    for markers, measure, title in [(density[:15], 'density', 'Woman−Man · densitas (1/2)'),
                                    (density[15:], 'density', 'Woman−Man · densitas (2/2)'),
                                    (binary, 'presence', 'Woman−Man · prevalensi')]:
        part = overall.loc[overall.marker_id.isin(markers)]
        bound = max(.001, overall.loc[overall.measurement.eq(measure), 'effect'].abs().max())
        unit = 'Selisih rerata / 100 kata' if measure == 'density' else 'Selisih prevalensi (poin persentase)'
        fig = plots.model_panels(part, markers, CONDITIONS, 'condition', 'effect', title,
                                 unit + '; * p Holm < 0,05.', limit=bound, diverging=True, stars=True)
        writer.add(fig, title, 'Positif: Woman lebih tinggi; negatif: Man lebih tinggi. Skala sama antar-model '
                   'dan antarhalaman dengan satuan yang sama. CI nominal, effect size, p Holm, n_pairs dan n_changed '
                   'tersedia di tabel. Tanpa bintang bukan bukti kesetaraan.', part)
    reference = pd.read_csv(reference_path, float_precision='round_trip')
    fig = plots.model_panels(reference, PERSONAS, INDICATORS, 'indicator', 'percent', 'Konteks rujukan gender',
                             '% respons tersedia; kategori dapat tumpang tindih.', row_key='persona',
                             labels=dict(zip(PERSONAS, plots.PERSONA_CODES)), column_labels=['Any', 'User', 'Hyp', '?', 'Scn'])
    writer.add(fig, 'Konteks rujukan gender', 'Any: penanda gender apa pun; User: rujukan pengguna eksplisit; '
               'Hyp: kutipan/hipotetis; ?: belum terklasifikasi; Scn: istilah sama dengan skenario, bukan bukti coreference. '
               'Klasifikasi otomatis candidate; kategori bertumpang tindih, bukan komposisi 100%.', reference)
    subset = subset_view(primary)
    markers = registry.marker_id.tolist()
    for i, selected in enumerate([markers[:21], markers[21:]], 1):
        part = subset.loc[subset.marker_id.isin(selected)].copy()
        part['symbol'] = [str(int(n)) if pd.notna(n) else ('Z' if s == 'zero_overall' else 'NA')
                          for n, s in zip(part.opposite_count, part.status)]
        fig = plots.grid_figure(part, selected, CONDITIONS, 'opposite_count', f'Peta arah antar-subset ({i}/2)',
                                '0–4: subset berlawanan tanda dengan overall; Z: overall nol.', symbols_column='symbol')
        writer.add(fig, f'Peta arah antar-subset ({i}/2)', 'Hitungan tanda efek dari commonsense, deontology, justice, virtue; '
                   'bukan uji signifikansi dan bukan rerata efek. Z = overall nol; NA = tidak tersedia. '
                   'Jumlah subset tersedia ada di tabel ringkasan; estimasi/CI seluruh subset ada di tables/comparisons.csv. '
                   'Nol pembalikan tanda tidak membuktikan robustness.', part)
    raw_direct = direct_rows(comparisons, registry, primary_only=False)
    normalization = normalization_view(raw_direct.loc[raw_direct.marker_id.isin(density)])
    mapping = {'same': 0, 'opposite': 1, 'one_zero': 2, 'both_zero': 3}
    for i, selected in enumerate([density[:15], density[15:]], 1):
        part = normalization.loc[normalization.marker_id.isin(selected)].copy()
        part['category'] = part.status.map(mapping)
        part['symbol'] = part.status.map({'same': '=', 'opposite': '↔', 'one_zero': 'Z', 'both_zero': '0', 'unavailable': 'NA'})
        fig = plots.grid_figure(part, selected, CONDITIONS, 'category', f'Jumlah mentah vs densitas ({i}/2)',
                                '= searah; ↔ berlawanan; Z: satu nol; 0: keduanya nol.', limit=3,
                                cmap=plots.NORMALIZATION_COLORS, symbols_column='symbol')
        fig.axes[-1].set_xticks([0, 1, 2, 3], ['=', '↔', 'Z', '0'])
        writer.add(fig, f'Jumlah mentah vs densitas ({i}/2)', 'Membandingkan tanda efek Woman−Man pada jumlah mentah '
                   'dan densitas per 100 kata. Tidak membandingkan besar efek lintas satuan dan bukan bukti kausal '
                   'pengaruh panjang respons. Tabel menyimpan kedua estimasi, CI, p Holm, dan perubahan keputusan Holm.', part)
    interaction = interaction_view(interactions, registry)
    writer.add(interaction_figure(interaction, registry), 'Perubahan selisih gender antar-skenario implisit',
               '1−2, 1−3, 2−3 berarti perbedaan selisih Woman−Man antara I1/I2/I3, memakai pasangan quadruplet '
               'yang sama dalam setiap perbandingan. Tidak ada perbandingan terhadap E dalam tabel interaksi sumber. '
               'Satuan densitas dan prevalensi dipisah; CI dan Holm disalin tanpa perubahan.', interaction)
    package(writer, registry, protected)
    print(f'Preview ready: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
