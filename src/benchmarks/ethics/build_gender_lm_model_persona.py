"""Build and install shared model-dot figures in the existing ETHICS notebook."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.text import Text
import nbformat
import numpy as np
import pandas as pd

from src.benchmarks.ethics import gender_lm_model_persona as plots
from src.benchmarks.ethics.gender_lm_compact_plots import overview_figure
from src.benchmarks.ethics.gender_lm_diagnostic_data import PERSONAS

SOURCE = ROOT / 'result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3'
TARGET = ROOT / 'notebooks/gender/linguistic marker/ETHICS'
NOTEBOOK = 'ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb'
REFERENCE = ROOT / 'result/ETHICS/reports/gender_lm_diagnostics_20260919_r1/diagnostic_tables/diagnostic_gender_reference_context.csv'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def relative(path):
    return Path(os.path.relpath(path, TARGET)).as_posix()


class Bundle:
    def __init__(self, path):
        if path.exists():
            raise FileExistsError('Build output must be a new path')
        (path / 'figures').mkdir(parents=True)
        self.path, self.index = path, []

    def add(self, fig, table, title, caption, audit=None):
        name = f'model_persona_{len(self.index) + 1:02d}'
        fig.canvas.draw()
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        assert min(t.get_fontsize() for t in texts) >= 8.5
        assert np.isclose(fig.get_figwidth() * 2.54, 16)
        assert fig.get_figheight() * 2.54 <= 26.01
        for suffix in ['png', 'svg']:
            fig.savefig(self.path / 'figures' / f'{name}.{suffix}', dpi=300)
        csv = self.path / 'figures' / f'{name}.csv'
        table.to_csv(csv, index=False)
        recovered = pd.read_csv(csv, float_precision='round_trip')
        assert len(table) == len(recovered)
        for col in table.select_dtypes(include=['floating']).columns:
            np.testing.assert_allclose(table[col], recovered[col], rtol=0, atol=0, equal_nan=True)
        record = dict(figure_id=name, title=title, caption=caption,
                      png_path=f'figures/{name}.png', svg_path=f'figures/{name}.svg',
                      source_csv=f'figures/{name}.csv', source_rows=len(table),
                      width_cm=16, height_cm=fig.get_figheight() * 2.54, min_font_pt=8.5,
                      audit=audit or {})
        self.index.append(record)
        plt.close(fig)
        print(f'{len(self.index)}/12 {title}', flush=True)


def package(bundle, inputs):
    intro = (
        '# Penanda linguistik Gender ETHICS — model dan persona\n\n'
        'Warna titik menunjukkan model; E/I1/I2/I3 menunjukkan skenario. '
        'Setiap panel memuat keenam model bersama. Semua 41 penanda tetap tercakup. '
        'Arah perbandingan mengikuti sumber ETHICS: Woman−Man, persona−Neutral, atau I−E, '
        'bukan membalik angka agar sama dengan arah GSM8K.\n\n'
        'Data penuh candidate r3; satu respons menjadi satu observasi. Penanda dihitung '
        'dari penjelasan final yang tersedia, termasuk abstain, tanpa reasoning channel. '
        'Missing/teknis bukan observasi nol; verdict-only memiliki penjelasan nol kata. '
        'Audit manusia belum selesai.\n\n'
        'Penanda density: titik adalah matched rank-biserial dari selisih densitas, '
        'bukan selisih rerata. CI rerata tidak dipasang pada rank-biserial. '
        'Penanda presence: titik adalah selisih prevalensi (poin persentase), dengan CI nominal 95%. '
        'Kepekatan memakai p Holm sumber; titik pudar bukan bukti kesetaraan. '
        'Ukuran titik mengikuti n_changed, bukan jumlah seluruh pasangan. '
        'Tanda × pada kolom NA berarti estimasi tidak tersedia, termasuk rank yang '
        'tidak terdefinisi ketika tidak ada pasangan berubah; bukan efek nol.\n\n'
        'Jumlah mentah, densitas per 100 kata, effect size, CI, p Holm, n_pairs, dan n_changed '
        'tetap tersedia dalam tabel. Hasil nonsignifikan tidak disaring. '
        'Tidak ada uji statistik baru atau pooled effect antarmodel.\n'
    )
    nb = nbformat.v4.new_notebook(metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                                         'language_info': {'name': 'python'}, 'presentation': 'shared-model-dots-v1'})
    nb.cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(
        'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n'
        f'NOTEBOOK_DIR = Path({str(TARGET)!r})\nSOURCE_DIR = Path({str(SOURCE)!r})\n'
        'def show_figure(name):\n    display(Image(filename=str(NOTEBOOK_DIR / "figures" / (name + ".png")), width=1100))\n'
        'def figure_table(name):\n    return pd.read_csv(NOTEBOOK_DIR / "figures" / (name + ".csv"), float_precision="round_trip")\n'
        'assert SOURCE_DIR.is_dir()\n')]
    pieces = []
    for i, record in enumerate(bundle.index, 1):
        title, caption, name = record['title'], record['caption'], record['figure_id']
        nb.cells.append(nbformat.v4.new_markdown_cell(f'## {i}. {title}\n\n{caption}'))
        output = nbformat.v4.new_output('display_data', data={
            'image/png': base64.b64encode((bundle.path / record['png_path']).read_bytes()).decode('ascii'),
            'text/plain': 'Pre-rendered figure from frozen source tables; run this cell to reload the local PNG.'})
        nb.cells.append(nbformat.v4.new_code_cell(f'show_figure({name!r})', outputs=[output], execution_count=None))
        nb.cells.append(nbformat.v4.new_markdown_cell(
            f'Gambar {i}. {title}.\n\n[Tabel lengkap]({record["source_csv"]}) · [PNG]({record["png_path"]}) · [SVG]({record["svg_path"]})'))
        pieces.append(f'<figure><img src="{record["png_path"]}" loading="lazy" alt="{html.escape(title)}">'
                      f'<figcaption><strong>Gambar {i}. {html.escape(title)}.</strong> {html.escape(caption)} '
                      f'<a href="{record["source_csv"]}">Tabel</a> · <a href="{record["svg_path"]}">SVG</a></figcaption></figure>')
    appendix = (
        '## Tabel lengkap dan lampiran\n\n'
        f'- [Jumlah mentah, densitas, dan prevalensi seluruh penanda]({relative(SOURCE / "marker_summary.csv")})\n'
        f'- [Seluruh perbandingan, termasuk empat subset]({relative(SOURCE / "comparisons.csv")})\n'
        f'- [Interaksi antarskenario]({relative(SOURCE / "interaction_comparisons.csv")})\n'
        f'- [Registry, sumber kamus, dan status adaptasi]({relative(SOURCE / "registry.json")})\n'
        '- [Galeri diagnostik lengkap](diagnostic_gallery.html)\n'
        '- [Indeks 88 grafik detail lama](figure_index.csv)\n'
        '- [Indeks 80 diagnostik lama](diagnostic_figure_index.csv)\n\n'
        'Grafik diagnostik lama tetap menyediakan CI selisih rerata, sensitivitas subset dan '
        'normalisasi, dukungan pasangan per marker, serta perbandingan implisit. '
        'Tabel panjang respons dan cakupan tersedia dalam CSV figur pertama. '
        'Ringkasan overall ini adalah penanda linguistik, bukan akurasi ETHICS gabungan. '
        'Jangan menyimpulkan bahwa selisih rank-biserial adalah selisih rerata atau '
        'bahwa p Holm menjadikan CI nominal sebagai interval simultan.\n')
    nb.cells.append(nbformat.v4.new_markdown_cell(appendix))
    nbformat.validate(nb)
    nbformat.write(nb, bundle.path / NOTEBOOK)
    intro_html = ''.join(f'<p>{html.escape(p)}</p>' for p in intro.split('\n\n'))
    footer = f'<p><a href="diagnostic_gallery.html">Diagnostik lengkap</a> · <a href="{relative(SOURCE / "marker_summary.csv")}">Jumlah mentah/densitas</a> · <a href="{relative(SOURCE / "comparisons.csv")}">Seluruh perbandingan</a></p>'
    page = ('<!doctype html><html lang="id"><meta charset="utf-8"><title>Gender ETHICS — model & persona</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:900px;margin:32px auto;padding:0 20px;color:#172033}'
            'figure{margin:48px 0}img{width:16cm;max-width:100%;height:auto;display:block;margin:auto}'
            'figcaption{max-width:16cm;margin:15px auto}a{color:#2456a6}</style><body>'
            + intro_html + ''.join(pieces) + footer + '</body></html>')
    (bundle.path / 'gender_ethics_model_persona.html').write_text(page, encoding='utf-8')
    pd.DataFrame([{k: v for k, v in r.items() if k != 'audit'} for r in bundle.index]).to_csv(bundle.path / 'model_persona_figure_index.csv', index=False)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    manifest = dict(statistics_bundle=str(SOURCE), target=str(TARGET), active_notebook=NOTEBOOK,
                    marker_count=41, model_count=6, main_figures=len(bundle.index),
                    inputs=inputs, figures=bundle.index, source_unchanged=True,
                    inference_recomputed=False, human_audit_complete=False,
                    generated_files={str(p.relative_to(bundle.path)): digest(p) for p in bundle.path.rglob('*') if p.is_file()})
    (bundle.path / 'model_persona_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def build(output):
    if output.exists():
        raise FileExistsError(output)
    names = ['registry.json', 'comparisons.csv', 'marker_summary.csv', 'report_status_coverage.csv', 'response_length_summary.csv']
    protected = [SOURCE / n for n in names] + [REFERENCE, TARGET / NOTEBOOK, Path(__file__),
                  Path(__file__).with_name('gender_lm_model_persona.py')]
    inputs = {str(p): digest(p) for p in protected}
    registry = pd.DataFrame(json.loads((SOURCE / 'registry.json').read_text(encoding='utf-8')))
    comparisons = pd.read_csv(SOURCE / 'comparisons.csv', float_precision='round_trip')
    lengths = pd.read_csv(SOURCE / 'response_length_summary.csv', float_precision='round_trip').query("scope == 'overall'")
    statuses = pd.read_csv(SOURCE / 'report_status_coverage.csv', float_precision='round_trip').query("scope == 'overall'")
    reference = pd.read_csv(REFERENCE, float_precision='round_trip')
    coverage = statuses.pivot(index=['model_id', 'persona'], columns='text_status', values='count').fillna(0)
    coverage['available_percent'] = 100 * (coverage.substantive_explanation + coverage.verdict_only) / coverage.sum(axis=1)
    coverage = coverage.reset_index()
    bundle = Bundle(output)
    bundle.add(overview_figure(coverage, lengths, PERSONAS), coverage.merge(lengths, on=['model_id', 'persona']),
               'Cakupan dan panjang respons', 'Cakupan sebelum membaca efek: tersedia mencakup verdict-only; missing/teknis bukan nol. '
               'Median, kuantil, dan jumlah setiap status tersedia dalam tabel. N = neutral, M/W = Man/Woman; '
               'L31/L33 = Llama 3.1/3.3, MIN = Ministral, DSR1 = DeepSeek R1, GPT = GPT-OSS, QWN = Qwen3 Next.')
    bundle.add(plots.persona_profile(lengths), lengths, 'Panjang respons: perbandingan model dan persona',
               'Warna menunjukkan model; Man dan Woman dipisahkan. Median mencakup verdict-only sebagai nol kata; '
               'missing/teknis dikecualikan. Ini profil deskriptif, bukan uji perbedaan panjang respons.')
    density = registry.loc[registry.measurement.eq('density'), 'marker_id'].tolist()
    binary = registry.loc[registry.measurement.eq('binary'), 'marker_id'].tolist()
    groups = [(density[:15], 'density', 'Numerik (1/2)'), (density[15:], 'density', 'Numerik (2/2)'),
              (binary, 'presence', 'Presence')]
    assert len(density) == 30 and len(binary) == 11
    captions = {'direct': 'Perbandingan langsung Woman−Man pada skenario yang sama.',
                'neutral': 'Baris panel memisahkan Man−Neutral dan Woman−Neutral; kolom memisahkan E/I1/I2/I3.',
                'explicit': 'Baris panel memisahkan persona Man dan Woman; kolom memisahkan I1−E, I2−E, I3−E.'}
    titles = {'direct': 'Woman−Man', 'neutral': 'Persona terhadap Neutral', 'explicit': 'Implisit terhadap Explicit'}
    for kind in ['direct', 'neutral', 'explicit']:
        selected = plots.select_contrasts(comparisons, registry, kind)
        expected = {'direct': 984, 'neutral': 1968, 'explicit': 1476}[kind]
        assert len(selected) == expected and selected.marker_id.nunique() == 41
        for markers, measurement, group_title in groups:
            table = selected.loc[selected.marker_id.isin(markers)].copy()
            title = titles[kind] + ' · ' + group_title
            fig, audit = plots.contrast_figure(table, markers, kind, measurement, title)
            caption = captions[kind] + ' Semua model berada pada sumbu yang sama; offset vertikal kecil hanya memisahkan titik.'
            caption += (' Rank-biserial berasal dari densitas per 100 kata. Nilai efek rerata dan CI-nya tetap ada dalam tabel/forest detail; '
                        'CI rerata tidak digunakan sebagai CI rank-biserial.' if measurement == 'density' else
                        ' Garis adalah CI nominal 95% selisih prevalensi, bukan interval simultan setelah Holm.')
            bundle.add(fig, table, title, caption, audit)
    bundle.add(plots.reference_figure(reference), reference, 'Konteks rujukan gender',
               'Warna model tetap sama; M/W menandai persona. Kategori otomatis dapat tumpang tindih. '
               'Istilah yang sama dengan skenario tidak membuktikan rujukan kepada pengguna atau adopsi identitas.')
    assert len(bundle.index) == 12
    package(bundle, inputs)
    print(f'Built: {output}', flush=True)


def install(output):
    manifest = json.loads((output / 'model_persona_manifest.json').read_text(encoding='utf-8'))
    assert all(digest(output / name) == h for name, h in manifest['generated_files'].items())
    assert all(digest(Path(path)) == h for path, h in manifest['inputs'].items())
    archive = TARGET / '_arsip' / 'sebelum_model_persona_20260920'
    destinations = [TARGET / name for name in manifest['generated_files'] if name != NOTEBOOK]
    destinations += [TARGET / 'model_persona_manifest.json', TARGET / 'model_persona_installation.json']
    if archive.exists() or any(p.exists() for p in destinations):
        raise FileExistsError('New presentation or backup already exists; do not overwrite')
    assert archive.resolve().is_relative_to((TARGET / '_arsip').resolve())
    existing_figures = {p: digest(p) for p in (TARGET / 'figures').iterdir() if p.is_file()}
    archive.mkdir()
    shutil.copy2(TARGET / NOTEBOOK, archive / NOTEBOOK)
    shutil.copy2(TARGET / 'README.md', archive / 'README.md')
    assert digest(archive / NOTEBOOK) == manifest['inputs'][str(TARGET / NOTEBOOK)]
    for name in manifest['generated_files']:
        if name != NOTEBOOK:
            shutil.copyfile(output / name, TARGET / name)
    shutil.copyfile(output / 'model_persona_manifest.json', TARGET / 'model_persona_manifest.json')
    shutil.copyfile(output / NOTEBOOK, TARGET / NOTEBOOK)
    assert all(digest(TARGET / name) == h for name, h in manifest['generated_files'].items())
    assert all(digest(p) == h for p, h in existing_figures.items())
    assert not list((TARGET / 'figures').glob('ringkas_*.png'))
    assert not list((TARGET / 'figures').glob('ringkas_*.csv'))
    verification = dict(active_notebook=str(TARGET / NOTEBOOK), backup=str(archive / NOTEBOOK),
                        installed_figures=12, previous_figure_files_unchanged=len(existing_figures),
                        source_statistics_unchanged=True, removed_ringkas_files_not_recreated=True)
    (TARGET / 'model_persona_installation.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--install-only', action='store_true')
    args = parser.parse_args()
    (install if args.install_only else build)(args.output.resolve())
