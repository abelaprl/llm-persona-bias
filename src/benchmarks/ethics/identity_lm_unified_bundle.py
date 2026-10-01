"""Offline paired gallery and executed-notebook content for identity markers."""
import html
from pathlib import Path

import nbformat
import pandas as pd

CONTRACT=[
    ('U01','Cakupan teks','GSM8K','Heatmap persona × model; jumlah dan persentase'),
    ('U02','Panjang respons','ETHICS','Satu panel persona; enam model; median dan IQR'),
    ('U03','Profil marker','ETHICS','Empat panel identitas; prevalence E; baris marker bersama'),
    ('U04','Perubahan dari netral','GSM8K + ETHICS','Empat panel identitas; E − N; titik model terpisah'),
    ('U05','Besar perbedaan dan CI','ETHICS','Titik dan CI sesuai estimand; bukan CI untuk rank'),
    ('U06','Kata dan ungkapan','GSM8K + ETHICS','Empat panel identitas; hit per 100 kata; enam model'),
    ('U07','Konteks kalimat','ETHICS','Cuplikan N/E/I1; model, identitas dan soal sama'),
]

CAUTION=('Bentuk bersama tidak menyamakan kamus, populasi, atau estimand antardataset. '
         'Hasil tetap candidate, bukan instrumen yang sudah tervalidasi. Seluruh teks tersedia '
         'pada sumber dipertahankan termasuk abstain, tanpa filter jawaban benar. Missing bukan nol. '
         'CI marginal bukan keputusan Holm; IQR bukan CI. Contoh kalimat bersifat purposif.')


def make_notebook(dimension, benchmark, records, output, notes):
    output=Path(output)
    selected=[r for r in records if r['benchmark']==benchmark]
    nb=nbformat.v4.new_notebook()
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                       language_info=dict(name='python'),figure_ids=[r['id'] for r in selected])
    nb.cells=[nbformat.v4.new_markdown_cell(
        f'# {dimension.title()} {benchmark} — template visual bersama\n\n'
        '[Galeri berpasangan](index.html) · [Metodologi dan reproduksi](README.md)\n\n'+CAUTION+'\n\n'
        '**U04/U05 GSM8K ditahan karena ketidakcocokan isi soal pada ID yang sama.** '
        '[Bukti dan batas pemakaian](PAIRING_AUDIT.md). Grafik kosong bukan efek nol.\n\n'
        'U01–U07 adalah cerita utama. Skenario implisit dan katalog seluruh marker berada di bagian detail. '
        'Kesamaan ID marker belum membuktikan definisinya identik.'),
        nbformat.v4.new_code_cell(
            'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n'
            f'OUTPUT = Path({output.as_posix()!r})\n'
            "if (Path.cwd() / 'visual_contract.csv').is_file():\n    OUTPUT = Path.cwd()\n"
            "display(pd.read_csv(OUTPUT / 'visual_contract.csv'))")]
    last=None
    for r in selected:
        if r['section']!=last:
            nb.cells.append(nbformat.v4.new_markdown_cell(
                '## Cerita utama' if r['section']=='main' else '## Skenario implisit dan katalog lengkap'))
            last=r['section']
        nb.cells += [nbformat.v4.new_markdown_cell(
            f"### {r['title']}\n\n{r['caption']}\n\n[PNG]({r['png']}) · [SVG]({r['svg']}) · [CSV]({r['csv']})"),
            nbformat.v4.new_code_cell(f"display(Image(filename=str(OUTPUT / {r['png']!r}), width=760))")]
        if r['pair_id'] in {'U04','U05'}:
            nb.cells.append(nbformat.v4.new_code_cell(
                f"support = pd.read_csv(OUTPUT / {r['csv']!r})\n"
                "cols = ['model_id','identity','scenario','row_id','estimate','changed','n_pairs','corrected_p']\n"
                "display(support[cols].sort_values(['changed','model_id']).head(12))"))
    b=benchmark.lower()
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## Kelengkapan dan sumber\n\n'
        f'- [Seluruh marker dan sumber kamus](tables/{b}_catalog.csv)\n'
        f'- [Seluruh profil dan jumlah mentah](tables/{b}_profile.csv)\n'
        f'- [Ringkasan kontras yang diadaptasi](tables/{b}_overview.csv)\n'
        f'- [Detail kontras yang diadaptasi](tables/{b}_detail.csv)\n'
        '- [Cakupan visual setiap marker](marker_coverage.csv)\n'
        '- [Verifikasi bentuk dan notebook](verification.json)\n'
        '- [Hash sumber sebelum/sesudah](source_hashes.json)\n\n'
        'Semua marker tercakup pada profil, tetapi tabel kontras hanya memuat estimand yang '
        'diadaptasi; statistik lain (termasuk binary GSM8K) tetap pada paket sumber yang ditautkan di galeri. '
        'Overall ETHICS di sini hanya merangkum marker, bukan akurasi lintas subset.'))
    return nb


def write_gallery(output, dimension, records, previous_links):
    esc=html.escape
    parts=['<!doctype html><html lang="id"><meta charset="utf-8">'
           f'<title>{esc(dimension.title())} — template bersama</title><style>'
           'body{font:16px system-ui;line-height:1.55;max-width:1450px;margin:30px auto;padding:0 20px}'
           '.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.card{min-width:0;padding:15px;background:#f8fafc}'
           'img{display:block;width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:30px;padding-top:15px}'
           'table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px;text-align:left}'
           'summary{cursor:pointer;font-weight:bold}a{color:#245ac0}'
           '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
           f'<h1>{esc(dimension.title())}: satu aturan visual, dua benchmark</h1>'
           '<p>GSM8K di kiri, ETHICS di kanan. Bentuk dipilih per jenis analisis, bukan per benchmark.</p>'
           f'<p>{esc(CAUTION)}</p><p><a href="GSM8K_unified.ipynb">Notebook GSM8K</a> · '
           '<a href="ETHICS_unified.ipynb">Notebook ETHICS</a> · <a href="README.md">Metodologi</a></p>'
           '<p style="padding:15px;background:#fff7ed;border:2px solid #c2410c">'
           '<strong>U04/U05 GSM8K ditahan.</strong> Ditemukan ID sama dengan isi soal berbeda. '
           'Panel kosong bukan efek nol. Kandidat visual ini tidak memperbaiki statistik sumber. '
           '<a href="PAIRING_AUDIT.md">Baca bukti dan batas pemakaian</a>.</p>']
    parts.append(pd.DataFrame(CONTRACT,columns=['Kode','Pertanyaan','Acuan desain','Bentuk bersama']).to_html(index=False))
    def card(r):
        return (f'<div class="card"><h3>{esc(r["benchmark"])}</h3><img loading="lazy" src="{r["png"]}" '
                f'alt="{esc(r["title"])}"><p>{esc(r["caption"])}</p>'
                f'<a href="{r["svg"]}">SVG</a> · <a href="{r["csv"]}">CSV</a></div>')
    for code,title,reference,shape in CONTRACT:
        rows=[r for r in records if r['pair_id']==code]
        if not rows:
            continue
        parts.append(f'<section id="{code}"><h2>{code} · {esc(title)}</h2><p>Acuan: {esc(reference)}. {esc(shape)}.</p><div class="pair">')
        for benchmark in ['GSM8K','ETHICS']:
            parts.append(card(next(r for r in rows if r['benchmark']==benchmark)))
        parts.append('</div></section>')
    parts.append('<section><h2>Detail skenario implisit</h2>')
    codes=list(dict.fromkeys(r['pair_id'] for r in records if r['section']=='detail' and not r['pair_id'].startswith('CAT')))
    for code in codes:
        rows=[r for r in records if r['pair_id']==code]
        parts.append(f'<details><summary>{esc(rows[0]["title"])}</summary><div class="pair">')
        for benchmark in ['GSM8K','ETHICS']:
            parts.append(card(next(r for r in rows if r['benchmark']==benchmark)))
        parts.append('</div></details>')
    parts.append('</section><section><h2>Katalog seluruh marker</h2><p>Katalog memakai prevalence E. '
                 'Seluruh persona dan statistik lain tetap ada di tabel lengkap; tidak diseleksi berdasarkan p-value.</p>')
    for r in records:
        if r['pair_id'].startswith('CAT'):
            parts.append(f'<details><summary>{esc(r["benchmark"]+" · "+r["title"])}</summary>'+card(r)+'</details>')
    for benchmark in ['gsm8k','ethics']:
        parts.append(f'<p>{benchmark.upper()}: <a href="tables/{benchmark}_catalog.csv">Kamus</a> · '
                     f'<a href="tables/{benchmark}_profile.csv">Semua profil</a> · '
                     f'<a href="tables/{benchmark}_overview.csv">Ringkasan yang diadaptasi</a> · '
                     f'<a href="tables/{benchmark}_detail.csv">Detail yang diadaptasi</a></p>')
    parts.append('<p>Semua marker tercakup pada profil; tabel kontras hanya memuat estimand yang '
                 'diadaptasi. Statistik lain, termasuk kontras binary GSM8K, tetap tersedia pada paket sumber di bawah.</p>')
    parts.append('<h3>Hasil sebelumnya tetap utuh</h3>')
    for label,target in previous_links:
        parts.append(f'<p><a href="{esc(target)}">{esc(label)}</a></p>')
    parts.append('</section></html>')
    (output/'index.html').write_text(''.join(parts),encoding='utf-8')
