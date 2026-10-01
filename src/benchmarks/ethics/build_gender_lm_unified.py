"""Build paired GSM8K/ETHICS figures with identical question-specific templates."""
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

from .gender_lm_reference_data import ROOT, digest
from .gender_lm_presentation_v2_data import MARKER_GROUPS
from .gender_lm_candidate_common import save_panel
from .gender_lm_unified_data import PREVIOUS, SHARED_MARKERS, PRIMARY, input_paths, load
from .gender_lm_unified_plots import (
    plot_coverage, plot_length, plot_profile, plot_effect, plot_terms, plot_examples,
    layout_signature, assert_matching_layouts,
)

DEFAULT_OUTPUT = PREVIOUS / 'unified_v1'
CONTRACT = [
    ('U01', 'Cakupan teks', 'GSM8K', 'Heatmap model × persona; jumlah dan persentase'),
    ('U02', 'Panjang respons', 'ETHICS', 'Satu panel persona; enam model; median dan IQR'),
    ('U03', 'Profil absolut', 'ETHICS', 'Panel N/Man E/Woman E; tujuh marker bersama'),
    ('U04', 'Ringkasan perbedaan', 'GSM8K + ETHICS', 'Lintas marker; titik model terpisah; panel 2×2 E/I1/I2/I3'),
    ('U05', 'Detail perbedaan', 'ETHICS', 'Titik + CI sesuai estimand; panel 2×2 E/I1/I2/I3'),
    ('U06', 'Kata dan ungkapan', 'GSM8K + ETHICS', 'Dua panel persona; enam model; kemunculan/100 kata'),
    ('U07', 'Konteks kalimat', 'ETHICS', 'Tiga cuplikan N/Man E/Woman E; model dan soal sama'),
]


def make_notebook(benchmark, records, output):
    output = Path(output)
    nb = nbformat.v4.new_notebook()
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                       language_info=dict(name='python'), unified_bundle_path=output.as_posix(),
                       figure_ids=[r['id'] for r in records])
    previous_link = Path(os.path.relpath(PREVIOUS, output)).as_posix().replace(' ', '%20')
    cells = [nbformat.v4.new_markdown_cell(
        f'# Gender {benchmark} — template visual bersama\n\n'
        'Setiap jenis analisis memakai **fungsi gambar yang sama** pada GSM8K dan ETHICS. '
        'Warna, posisi model, susunan panel, dan urutan persona bukan lagi dua desain terpisah.\n\n'
        '[Galeri berpasangan](index.html) · [Kontrak visual dan reproduksi](README.md)\n\n'
        'Hasil tetap candidate; kamus belum menjadi instrumen tervalidasi. Seluruh teks yang tersedia '
        'tetap masuk, termasuk abstain, tanpa filter jawaban benar atau hidden reasoning. '
        'Missing tidak diimputasi menjadi nol. Tujuh marker bersama dipilih untuk menyamakan '
        'urutan pembacaan, bukan karena p-value; kesamaan nama bukan bukti validitas lintas benchmark.\n\n'
        'Sumbu dan interval tetap mengikuti statistik sumber: U04 GSM8K rank-biserial tanpa CI rank; '
        'ETHICS delta rerata density tanpa CI pada ringkasan. U05 GSM8K delta median berpasangan, ETHICS delta rerata '
        'berpasangan. Keduanya bukan estimand yang sama. CI marginal bukan keputusan Holm.\n\n'
        'U06 memakai jumlah kemunculan dibagi total kata, bukan rata-rata density per respons. '
        'Kamus kata GSM8K dan frasa candidate ETHICS tidak identik; grafik ini deskriptif.\n\n'
        f'[Seluruh analisis tambahan pada candidate sebelumnya]({previous_link}/index.html) tetap tersedia.'),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
            f"OUTPUT = Path({output.as_posix()!r})\n"
            "if (Path.cwd() / 'visual_contract.csv').is_file():\n    OUTPUT = Path.cwd()\n"
            "display(pd.read_csv(OUTPUT / 'visual_contract.csv'))")]
    last = None
    for row in records:
        if row['section'] != last:
            cells.append(nbformat.v4.new_markdown_cell(
                '## Cerita utama' if row['section'] == 'main' else '## Detail lengkap dan pemeriksaan tambahan'))
            last = row['section']
        cells.append(nbformat.v4.new_markdown_cell(
            f"### {row['title']}\n\n{row['caption']}\n\n"
            f"[PNG]({row['png']}) · [SVG]({row['svg']}) · [CSV]({row['csv']})"))
        cells.append(nbformat.v4.new_code_cell(
            f"display(Image(filename=str(OUTPUT / {row['png']!r}), width=760))"))
        if row['pair_id'] in {'U04', 'U05'}:
            cells.append(nbformat.v4.new_code_cell(
                f"support = pd.read_csv(OUTPUT / {row['csv']!r})\n"
                "columns = ['model_id', 'panel', 'marker_id', 'estimate', 'changed', 'n_pairs', 'corrected_p']\n"
                "display(support[columns].sort_values(['changed', 'model_id']).head(12))"))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Kelengkapan dan sumber\n\n'
        f'- [Seluruh kontras direct]({benchmark.lower()}_all_effects.csv)\n'
        f'- [Seluruh profil marker dan jumlah mentah]({benchmark.lower()}_all_profiles.csv)\n'
        '- [Cakupan visual marker](marker_coverage.csv)\n'
        '- [Bentuk chart benar-benar cocok](layout_checks.json)\n'
        '- [Hash sumber sebelum/sesudah](source_hashes.json)\n\n'
        'Sebaran IQR bukan CI. CI bukan bukti signifikansi terkoreksi; Holm tetap pada CSV. '
        'Overall ETHICS hanya merangkum marker, bukan akurasi lintas subset. '
        'Contoh kalimat purposif, bukan sampel representatif; referen gender dapat berasal dari tokoh soal.'))
    nb.cells = cells
    return nb


def write_gallery(output, records):
    parts = ['<!doctype html><html lang="id"><meta charset="utf-8">'
             '<title>Template bersama — Gender GSM8K dan ETHICS</title><style>'
             'body{font:16px system-ui;line-height:1.55;max-width:1450px;margin:30px auto;padding:0 20px}'
             '.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.card{min-width:0;background:#f8fafc;padding:18px}'
             'img{display:block;width:100%;height:auto}section{border-top:1px solid #ccd;margin-top:30px;padding-top:15px}'
             'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:left}'
             'a{color:#245ac0}summary{cursor:pointer;font-weight:bold}'
             '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
             '<h1>Satu aturan visual, dua benchmark</h1><p>GSM8K di kiri, ETHICS di kanan. '
             'Untuk setiap pertanyaan, kedua gambar memakai template dan pemetaan model yang sama. '
             'Satuan, estimand, dan populasi tetap dijelaskan; ini bukan penyamaan instrumen statistik.</p>'
             '<p><a href="GSM8K_unified.ipynb">Notebook GSM8K</a> · '
             '<a href="ETHICS_unified.ipynb">Notebook ETHICS</a> · '
             '<a href="README.md">Catatan metodologi</a></p>']
    parts.append(pd.DataFrame(CONTRACT, columns=['Kode', 'Pertanyaan', 'Acuan desain', 'Bentuk bersama']).to_html(index=False))
    for code, question, reference, shape in CONTRACT:
        parts.append(f'<section id="{code}"><h2>{code} · {html.escape(question)}</h2>'
                     f'<p>Acuan: {html.escape(reference)}. {html.escape(shape)}.</p><div class="pair">')
        for benchmark in ['GSM8K', 'ETHICS']:
            row = next(r for r in records if r['benchmark'] == benchmark and r['pair_id'] == code)
            parts.append(f'<div class="card"><h3>{benchmark}</h3><img loading="lazy" '
                         f'src="{row["png"]}" alt="{html.escape(row["title"])}">'
                         f'<p>{html.escape(row["caption"])}</p><a href="{row["svg"]}">SVG</a> · '
                         f'<a href="{row["csv"]}">CSV</a></div>')
        parts.append('</div></section>')
    parts.append('<section><h2>Detail lengkap</h2>')
    for row in [r for r in records if r['section'] == 'detail']:
        parts.append(f'<details><summary>{row["benchmark"]} · {html.escape(row["title"])}</summary>'
                     f'<p>{html.escape(row["caption"])}</p><img loading="lazy" src="{row["png"]}" '
                     f'style="max-width:700px" alt="{html.escape(row["title"])}">'
                     f'<p><a href="{row["csv"]}">CSV</a></p></details>')
    parts.append('</section></html>')
    (output / 'index.html').write_text(''.join(parts), encoding='utf-8')


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Gunakan folder baru; tidak menimpa {output}')
    from . import gender_lm_unified_evidence as evidence
    sources = set(input_paths() + evidence.input_paths())
    protected = sorted(sources | {p for p in PREVIOUS.rglob('*') if p.is_file()})
    print('[1/5] Mencatat sumber dan candidate sebelumnya', flush=True)
    before = {str(p): digest(p) for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    for folder in ['figures', 'tables']:
        (output / folder).mkdir()
    status = dict(status='building', bundle='gender-shared-templates-v1',
                  created_at=datetime.now(timezone.utc).isoformat(), source_paths=[str(p) for p in sorted(sources)])
    def manifest():
        (output / 'manifest.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest()
    try:
        records, coverage = [], []
        def export(benchmark, code, title, caption, fig, table, section='main'):
            layout = layout_signature(fig)
            record = save_panel(output, benchmark.lower()+'_'+code, title, title, caption, fig, table, section)
            record.update(benchmark=benchmark, pair_id=code, layout=layout)
            records.append(record)
            return record
        print('[2/5] Merender satu template untuk setiap pasangan analisis', flush=True)
        for benchmark in ['GSM8K', 'ETHICS']:
            data = load(benchmark)
            gsm = benchmark == 'GSM8K'
            print(f'  {benchmark}: tujuh gambar utama', flush=True)
            export(benchmark, 'U01', 'U01 · Cakupan teks',
                   'Heatmap model × persona; angka = teks tersedia dan persentase terhadap target. '
                   + ('GSM8K memakai visible_text yang tersedia.' if gsm else
                      'ETHICS memakai penjelasan substantif; verdict-only tetap diinventarisasi, bukan nol marker.'),
                   plot_coverage(data['coverage'], benchmark, 'Angka = jumlah dan persentase terhadap target.\n'
                                 'M = Man; W = Woman; missing tidak menjadi nol marker.'), data['coverage'])
            export(benchmark, 'U02', 'U02 · Panjang respons',
                   'Satu panel yang sama: persona di sumbu vertikal, enam model berwarna, median dan IQR. '
                   'Rentang sumbu mengikuti data tiap benchmark. '
                   + ('GSM8K memakai seluruh visible_text tersedia.' if gsm else
                      'ETHICS mempertahankan verdict-only sebagai 0 kata sesuai sumber; denominator ini berbeda dari marker.'),
                   plot_length(data['length'], benchmark, 'Titik = median; garis = IQR, bukan CI.\n' +
                               ('Teks tersedia; tanpa filter jawaban benar.' if gsm else
                                'Termasuk verdict-only (0 kata); tanpa missing/technical-only.')), data['length'])
            export(benchmark, 'U03', 'U03 · Profil absolut marker',
                   'Tujuh marker bersama memakai urutan baris identik. Profil N/Man E/Woman E adalah '
                   'prevalence seluruh teks tersedia, bukan pasangan soal. Semua profil persona, raw count '
                   'dan density tetap disimpan dalam tabel lengkap; kesamaan ID bukan validasi instrumen.',
                   plot_profile(data['profile'], SHARED_MARKERS, benchmark,
                                '7 marker bersama; seluruh teks tersedia, bukan pasangan.\n'
                                'Profil deskriptif; raw count dan denominator pada CSV.'), data['profile'])
            export(benchmark, 'U04', 'U04 · Ringkasan perbedaan persona',
                   'Template 2×2 E/I1/I2/I3 dan urutan tujuh marker sama; titik enam model dipisah. '
                   'Titik kosong berarti <20 pasangan berubah, bukan keputusan signifikansi. '
                   + ('GSM8K mempertahankan rank-biserial tanpa CI rank. Nilai ekstrem pada sedikit '
                      'pasangan berubah tidak membuktikan perubahan luas.' if gsm else
                      'ETHICS mempertahankan delta rerata density; CI ditampilkan pada U05. '
                      'Nilainya tidak dibandingkan langsung dengan rank-biserial GSM8K.'),
                   plot_effect(data['overview'], SHARED_MARKERS, benchmark, 'U04',
                               'Rank-biserial Man − Woman' if gsm else 'Delta rerata (kemunculan/100 kata)',
                               'Man − Woman; titik kosong: <20 pasangan berubah.\n'
                               + ('Rank-biserial tanpa CI rank; dukungan/Holm: CSV.' if gsm else
                                  'Ringkasan tanpa CI; lihat U05 untuk CI dan CSV untuk Holm.'),
                               rank=gsm, intervals=False), data['overview'])
            export(benchmark, 'U05', 'U05 · Besar perbedaan dan CI',
                   'Bentuk forest plot sama dengan estimand dinyatakan eksplisit: '
                   + ('GSM8K memakai median selisih density berpasangan beserta CI median sumber.' if gsm else
                      'ETHICS memakai rerata selisih density berpasangan beserta CI rerata sumber.')
                   + ' Median dan rerata bukan estimand yang sama, walaupun satuannya sama. '
                   'CI marginal tidak menggantikan Holm/Wilcoxon sumber; angka nol tetap dipertahankan.',
                   plot_effect(data['detail'], SHARED_MARKERS, benchmark, 'U05',
                               'Delta median (kemunculan/100 kata)' if gsm else 'Delta rerata (kemunculan/100 kata)',
                               ('GSM8K: delta median; garis = CI bootstrap 95%.\n' if gsm else
                                'ETHICS: delta rerata; garis = CI bootstrap 95%.\n')+
                               'Man − Woman; titik kosong: <20 pasangan berubah.\n'
                               'CI marginal, bukan terkoreksi Holm.'), data['detail'])
            terms = evidence.expression_data(benchmark.lower())
            examples = evidence.example_data(benchmark.lower())
            notes = evidence.evidence_notes(benchmark.lower())
            export(benchmark, 'U06', 'U06 · Kata dan ungkapan teramati', notes['expressions'],
                   plot_terms(terms, benchmark, 'Hit dibagi total kata; bukan rerata density per respons.\n'
                              'Pilihan deskriptif; kata dapat merujuk tokoh soal.'), terms)
            export(benchmark, 'U07', 'U07 · Konteks kalimat', notes['examples'],
                   plot_examples(examples, benchmark, 'Ilustrasi purposif, bukan sampel representatif.\n'
                                 'Teks utuh dan provenance: CSV; referen gender belum diaudit.'), examples)
            data['effects'].to_csv(output / f'{benchmark.lower()}_all_effects.csv', index=False)
            data['all_profiles'].to_csv(output / f'{benchmark.lower()}_all_profiles.csv', index=False)
            if gsm:
                groups = [('D01', PRIMARY[:8]), ('D02', PRIMARY[8:])]
            else:
                groups = list(MARKER_GROUPS.items())
            for group, markers in groups:
                rows = data['effects'].loc[data['effects'].marker_id.isin(markers)].copy()
                unit = rows.display_unit.iloc[0]
                xlabel = ('Rank-biserial Man − Woman' if gsm else
                          'Man − Woman (poin persentase)' if unit == 'percentage_points' else
                          'Delta rerata (kemunculan/100 kata)')
                record = export(benchmark, 'detail_'+group, f'{group} · Detail semua indikator kelompok',
                                'Semua indikator dipertahankan, tanpa seleksi p-value. Template panel dan posisi '
                                'model sama dengan ringkasan. Dukungan, Holm dan satuan sumber pada CSV.',
                                plot_effect(rows, markers, benchmark, group, xlabel,
                                            'Man − Woman; titik kosong: <20 pasangan berubah.\n'
                                            + ('Rank-biserial tanpa CI rank; dukungan/Holm: CSV.' if gsm else
                                               'Garis = CI 95% marginal; dukungan/Holm: CSV.'), rank=gsm), rows, 'detail')
                coverage.extend(dict(benchmark=benchmark, marker_id=m, group=group, figure=record['png'],
                                     shown_in_main=m in SHARED_MARKERS) for m in markers)
            if not gsm:
                zoom = data['effects'].loc[data['effects'].marker_id.isin(MARKER_GROUPS['B08']) &
                                           data['effects'].panel.ne('E')].copy()
                export(benchmark, 'detail_B08_zoom', 'B08 · Pembesaran implisit',
                       'I1/I2/I3 diperbesar pada satu skala bersama; tidak ada angka berubah. '
                       'Bandingkan B08 lengkap untuk skala yang juga mencakup E.',
                       plot_effect(zoom, MARKER_GROUPS['B08'], benchmark, 'B08 zoom',
                                   'Man − Woman (poin persentase)',
                                   'SKALA DIPERBESAR: hanya I1–I3; lihat B08 lengkap.\n'
                                   'CI 95% marginal; dukungan dan Holm pada CSV.', panels=['I1','I2','I3']), zoom, 'detail')
        print('[3/5] Membandingkan geometri pasangan dan menulis galeri', flush=True)
        checks = {}
        for code, _, _, _ in CONTRACT:
            paired = [r for r in records if r['pair_id'] == code]
            if len(paired) != 2:
                raise ValueError('Missing comparison figure')
            assert_matching_layouts(paired[0]['layout'], paired[1]['layout'])
            checks[code] = dict(matching=True, signature=paired[0]['layout'])
        (output / 'layout_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        pd.DataFrame(CONTRACT, columns=['pair_id', 'question', 'visual_reference', 'shared_template']).to_csv(
            output / 'visual_contract.csv', index=False)
        pd.DataFrame(coverage).to_csv(output / 'marker_coverage.csv', index=False)
        pd.DataFrame(records).drop(columns=['geometry', 'layout']).to_csv(output / 'figure_index.csv', index=False)
        (output / 'figure_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        write_gallery(output, records)
        old_link = Path(os.path.relpath(PREVIOUS, output)).as_posix().replace(' ', '%20')
        (output / 'README.md').write_text(
            '# Gender — satu aturan visual untuk dua benchmark\n\n'
            '[Buka galeri berpasangan](index.html). [GSM8K](GSM8K_unified.ipynb) · [ETHICS](ETHICS_unified.ipynb).\n\n'
            '## SPEC dan DESIGN\n\n'
            'Perubahan hanya penyajian. Seluruh respons yang tersedia pada sumber tetap menjadi populasi; '
            'tidak ada sampling baru untuk statistik, API call, ekstraksi, kamus, atau pengujian inferensial baru. '
            'Tujuh jenis gambar utama memakai fungsi yang sama, bukan implementasi berbeda per benchmark. '
            'Galeri memasangkan hasilnya agar kesamaan bentuk dapat diperiksa langsung.\n\n'
            '[Kontrak pemilihan chart](visual_contract.csv): coverage mengambil heatmap GSM8K; panjang respons '
            'mengambil satu panel ETHICS; ringkasan menggabungkan cakupan lintas marker GSM8K dan pemisahan titik '
            'model serta panel 2×2 ETHICS; detail memakai CI yang sesuai; kata/ungkapan disertai konteks.\n\n'
            '## IMPLEMENTATION dan batas statistik\n\n'
            '- Font DejaVu Sans minimal 9 pt; lebar gambar 16 cm. PNG 240 dpi dan SVG tersedia.\n'
            '- U03/U04/U05 memakai tujuh marker bersama yang ditetapkan sebelum pemilihan berdasarkan data. '
            'Semua 16 indikator numerik GSM8K dan 41 marker ETHICS tetap ada pada detail dan marker_coverage.csv.\n'
            '- U04: titik tanpa CI untuk keduanya; GSM8K rank-biserial, ETHICS delta rerata density. '
            'Susunan sama tidak berarti angkanya sebanding. CI ditampilkan pada U05.\n'
            '- U05: GSM8K median selisih density, ETHICS rerata selisih density. CI milik estimand masing-masing; '
            'tidak memakai CI median untuk rank, tidak mengubah median menjadi rerata.\n'
            '- CI marginal tidak sama dengan keputusan Holm. Titik kosong menunjukkan <20 pasangan berubah, '
            'bukan signifikansi; semua pasangan, p-value dan effect size tetap ada di CSV.\n'
            '- U02 ETHICS mempertahankan verdict-only sebagai 0 kata. Denominator panjang respons berbeda dari '
            'denominator penjelasan substantif pada marker; missing tidak menjadi nol.\n'
            '- U06 jumlah hit dibagi total kata ×100, dengan raw count, total_words dan n_available disimpan. '
            'Ini pooled density kata, bukan mean density respons. Pemilihan istilah deskriptif, bukan uji baru.\n'
            '- Kamus/surface berbeda: GSM8K kata pada visible_text, ETHICS frasa candidate pada explanation_text. '
            'Referen dapat berasal dari soal. Semua instrumen tetap candidate, belum selesai audit manual.\n'
            '- U07 triplet purposif: model/soal sama di dalam benchmark, bukan soal sama lintas benchmark. '
            'Teks penuh, lokasi source dan offset cuplikan disimpan; contoh bukan sampel representatif.\n\n'
            f'[Candidate sebelumnya beserta opening, pronomina, perbandingan netral/implisit, kamus dan bukti]({old_link}/index.html) '
            'tetap utuh. Tidak ada berkas lama dihapus.\n\n'
            '## CHECKLIST dan TESTING\n\n'
            '- [x] Fungsi plot bersama dan geometri ketujuh pasangan dicocokkan.\n'
            '- [x] Seluruh indikator sumber tetap tercakup; raw count/denominator tersedia.\n'
            '- [x] PNG/SVG/CSV dibuat pada folder baru; angka inferensial sumber tidak dihitung ulang.\n'
            '- [x] Kedua notebook dieksekusi; hasil tercatat di verification.json.\n'
            '- [x] Sumber dan candidate sebelumnya diperiksa dengan hash sebelum/sesudah.\n\n'
            '## INTEGRATION dan reproduksi\n\n'
            'Gunakan galeri untuk membandingkan jenis visualisasi terlebih dahulu. Belum mengganti notebook utama. '
            'Dari D:\\TugasAkhir, gunakan **path baru** untuk membangun ulang:\n\n'
            '```powershell\npython -m src.benchmarks.ethics.build_gender_lm_unified '
            '--output "notebooks/gender/linguistic marker/candidate/unified_rerun"\n```\n', encoding='utf-8')
        print('[4/5] Mengeksekusi kedua notebook', flush=True)
        notebook_checks = {}
        for benchmark in ['GSM8K', 'ETHICS']:
            selected = [r for r in records if r['benchmark'] == benchmark]
            notebook = make_notebook(benchmark, selected, output)
            NotebookClient(notebook, timeout=180, kernel_name='python3',
                           resources={'metadata': {'path': str(output)}}).execute()
            errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
            pngs = sum('image/png' in o.get('data', {}) for c in notebook.cells if c.cell_type == 'code' for o in c.outputs)
            if errors or pngs != len(selected):
                raise ValueError(f'Incomplete notebook: {benchmark}')
            nbformat.write(notebook, output / f'{benchmark}_unified.ipynb')
            notebook_checks[benchmark] = dict(errors=0, embedded_pngs=pngs)
        print('[5/5] Memastikan sumber dan candidate lama tetap utuh', flush=True)
        after = {str(p): digest(p) for p in protected}
        if before != after:
            raise ValueError('A protected input or previous artifact changed')
        (output / 'source_hashes.json').write_text(json.dumps(dict(before=before, after=after, unchanged=True), indent=2), encoding='utf-8')
        code_names = ['build_gender_lm_unified.py', 'gender_lm_unified_data.py', 'gender_lm_unified_plots.py',
                      'gender_lm_unified_evidence.py', 'gender_lm_candidate_common.py', 'gender_lm_reference_data.py',
                      'gender_lm_reference_plots.py', 'gender_lm_candidate_gsm8k.py', 'gender_lm_presentation_v2_data.py',
                      'gender_lm_presentation_v2_plots.py', 'restyle_gender_lm_gsm8k.py']
        code_paths = [Path(__file__).with_name(n) for n in code_names]
        (output / 'code_hashes.json').write_text(json.dumps({str(p): digest(p) for p in code_paths}, indent=2), encoding='utf-8')
        verification = dict(notebooks=notebook_checks, matching_pairs=len(checks), figures=len(records),
                            protected_files=len(before), old_files_unchanged=True,
                            marker_coverage=pd.DataFrame(coverage).groupby('benchmark').marker_id.nunique().to_dict(),
                            min_font_pt=min(r['geometry']['min_font_pt'] for r in records))
        (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        status['status'] = 'complete_candidate_presentation'
        manifest()
        return verification
    except Exception as error:
        status.update(status='incomplete_do_not_use', error=str(error))
        manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == '__main__':
    main()
