"""Portable notebook links and reviewed navigation; immutable report copies retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def portable_links(text,source_directory,destination_directory):
    def replace(match):
        link=match.group(1)
        if link.startswith(('https://','http://','#')):return match.group(0)
        normalized=link.replace('\\','/')
        target=(source_directory/normalized).resolve()
        if not target.is_file():raise FileNotFoundError(target)
        relative=os.path.relpath(target,destination_directory).replace('\\','/')
        return ']('+relative+')'
    return re.sub(r'\]\(([^)]+)\)',replace,text)


def finalize(report):
    import nbformat
    from src.benchmarks.ethics.religion_lm_v3_io import write_json,sha256
    manifest=json.loads((report/'manifest.json').read_text(encoding='utf-8'))
    source=report/manifest['default_notebook'];destination=Path(manifest['notebook_path'])
    expected=ROOT/'notebooks/religion'/('linguistic_marker_religion_ethics_v3_candidate_'+report.name.removeprefix('religion_lm_v3_candidate_')+'.ipynb')
    assert destination==expected and destination.is_file()
    reviewed=report/'religion_ethics_linguistic_markers_v3_reviewed_candidate.ipynb'
    if reviewed.exists() or (report/'presentation_review_manifest.json').exists():raise FileExistsError(reviewed)
    notebook=nbformat.read(source,as_version=4)
    for cell in notebook.cells:cell.source=portable_links(cell.source,report,report)
    intro='## Membaca hasil kandidat\n\n154.428 respons, 59 marker, enam model dan 17 persona; empat subset digabung untuk marker. Sebanyak 46.052 respons abstain berteks tetap masuk analisis. Ini belum precision/recall tervalidasi.\n\n'
    for title,path in [('Laporan hasil dan keterbatasan',ROOT/'docs/religion_lm_v3_results.md'),
        ('Definisi dan seluruh pola marker',ROOT/'docs/religion_lm_v3_codebook.md'),
        ('Paket precision held-out independen',ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_precision_heldout_20260916_r1/manifest.json'),
        ('Paket recall tanpa anchor',ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_human_audit_20260916_r2/manifest.json')]:
        intro+='['+title+']('+os.path.relpath(path,report).replace('\\','/')+')\n\n'
    intro+='Pakai PDF A4 untuk ekspor tesis. Forest enam model adalah overview; plot implicit memakai I1 − explicit. I2/I3 tetap tersedia terpisah pada penjelajah. Marker lexical mention tidak otomatis menunjukkan adopsi agama atau endorsement. Sembilan gap discovery masih terbuka.\n'
    cell=nbformat.v4.new_markdown_cell(intro);cell.id='reviewed-reading-guide'
    notebook.cells.insert(1,cell)
    nbformat.validate(notebook)
    with reviewed.open('x',encoding='utf-8') as handle:nbformat.write(notebook,handle)
    for cell in notebook.cells:cell.source=portable_links(cell.source,report,destination.parent)
    # This is the new notebook created in this task, outside immutable result/.
    with destination.open('w',encoding='utf-8') as handle:nbformat.write(notebook,handle)
    readme=destination.parent/'README.md'
    existing=readme.read_text(encoding='utf-8')
    previous_hash=hashlib.sha256(existing.encode('utf-8')).hexdigest()
    content='# Notebook Religion yang dipakai\n\nDefault linguistic marker ETHICS: **kandidat v3, keseluruhan ETHICS**. Enam model dan kondisi E/I1/I2/I3 tetap terpisah; tidak ada akurasi ETHICS gabungan.\n\n'
    content+='[Buka notebook v3]('+destination.name+') · [Laporan hasil](../../docs/religion_lm_v3_results.md)\n\n'
    prefix=os.path.relpath(report,readme.parent).replace('\\','/')+'/'
    content+='[PDF A4 — 24 panel per model]('+prefix+'religion_ethics_marker_publication_panels.pdf) · [Semua 52 kontras/model]('+prefix+'evidence/all_contrasts_explorer.html) · [37 PNG/SVG]('+prefix+'figures/)\n\n'
    content+='59 marker mencakup delapan keluarga. Semua marker masih candidate; anotasi dua reviewer dan precision/recall empiris belum tersedia. Baca denominator, status CI dan gap pada laporan sebelum menarik kesimpulan.\n\n'
    content+='## Pembanding dan analisis lain\n\n| Kebutuhan | Buka |\n|---|---|\n'
    content+='| Marker ETHICS sebelumnya | [Notebook v2.0.2](linguistic_marker_religion_ethics_thesis_v2_0_2_candidate_20260827.ipynb) |\n'
    content+='| Atlas sumber v2.0.2 | [Atlas](linguistic_marker_religion_ethics_atlas_v2_0_2_candidate_20260827.ipynb) |\n'
    content+='| Akurasi ETHICS | [Notebook akurasi](<ethics accuracy analysis/ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb>) |\n'
    content+='| Akurasi GSM8K | [Notebook akurasi](<akurasi analysis/GSM8K_Religion_Accuracy_Analysis_Lengkap.ipynb>) |\n\n'
    content+='Notebook GSM8K dan hasil marker lama dipertahankan. Penyajian September sebelumnya juga sudah overall; v3 memperluas deteksi dan audit. `img/ethics/` tetap aset historis; gambar kandidat baru berada pada folder hasil v3 yang ditautkan di atas.\n'
    readme.write_text(content,encoding='utf-8')
    write_json(report/'presentation_review_manifest.json',dict(preferred_report_notebook=reviewed.name,workspace_notebook=str(destination),
        report_notebook_sha256=sha256(reviewed),workspace_notebook_sha256=sha256(destination),readme_sha256=sha256(readme),
        previous_readme_text_sha256=previous_hash,source_notebook_retained=str(source),
        supersedes_default_notebook_navigation_only=True,numeric_data_unchanged=True,portable_local_links_checked=True,
        script_sha256=sha256(__file__)))
    print('Reviewed notebooks and README navigation written.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True,type=Path)
    finalize(parser.parse_args().report.resolve())
