"""Executed, relocatable notebook consuming frozen Regional artifacts."""
import os
from pathlib import Path
import re
from urllib.parse import unquote
import nbformat
import pandas as pd
from .regional_ethics_presentation_data import ROOT, POLICY, sha256

NOTEBOOK_NAME = 'linguistic_marker_regional_cultural_ethics_v3_candidate.ipynb'


def resolve_figure(bundle, document_dir, relative, expected):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe figure path')
    for base in [Path(document_dir), Path(bundle)]:
        path = base / relative
        if path.is_file() and sha256(path) == expected:
            return path
    raise ValueError(f'No hash-matching image: {relative}')


def make_notebook(bundle, document_dir, published_bundle=None):
    bundle, document_dir = Path(bundle), Path(document_dir)
    published = Path(published_bundle or bundle)
    index = pd.read_csv(bundle / 'figure_index.csv')
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# Linguistic marker Regional/Cultural — ETHICS v3

**Candidate — belum lolos audit dua reviewer manusia.** Identitas regional di sini adalah persona eksperimen, bukan sifat penduduk wilayah tersebut.

Sumber: Regional v3 r5; 154.428 respons, 6 model, 17 persona, 53 marker (23 regional). Pengukuran memakai final surface, bukan hidden reasoning. Respons label-only/teknis/missing bukan penjelasan dengan nol marker. *Abstain* berpenjelasan tetap masuk.

Policy: `regional-explanation-v1-pairing-v2-20260919`. Statistik berpasangan dihitung ulang dari respons eligible, tidak mengambil p-value lama. Satu respons = satu observasi; N yang berulang di panel tetap observasi yang sama.

**Pengecualian pairing yang disetujui:** 20 respons DeepSeek neutral (ID 0001–0005 pada masing-masing subset) memakai soal berbeda. Seluruhnya tetap disimpan dalam coverage dan deskriptif, tetapi tidak boleh membentuk pasangan inferensial. Daftar kunci dan bukti request tersedia pada tabel pengecualian. Pengecualian ini tidak mengubah scoring historis; akurasi historis kelompok terdampak belum dapat dianggap tervalidasi karena mismatch soal.

GPT-OSS memiliki sangat sedikit pasangan eligible (12–71 utama dan 2–20 sensitivitas sebelum pemeriksaan NA per outcome). Selalu baca `n_pairs`; besar korpus awal bukan ukuran dukungan setiap kontras.
''')]
    location = document_dir.relative_to(ROOT).as_posix()
    cells.append(code(f'''from pathlib import Path
import os, sys, json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/MAIN_PARSING/ETHICS').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.MAIN_PARSING.ETHICS.regional_ethics_presentation_notebook import resolve_figure
PUBLISHED = ROOT / {published.relative_to(ROOT).as_posix()!r}
BUNDLE = PUBLISHED if PUBLISHED.is_dir() else ROOT / {bundle.relative_to(ROOT).as_posix()!r}
DOCUMENT_DIR = ROOT / {location!r}
if DOCUMENT_DIR == ROOT / {bundle.relative_to(ROOT).as_posix()!r}: DOCUMENT_DIR = BUNDLE
index = pd.read_csv(BUNDLE / 'figure_index.csv')
def link(relative, label):
    target = os.path.relpath(BUNDLE / relative, DOCUMENT_DIR).replace('\\\\', '/')
    display(Markdown('[' + label + '](<' + target + '>)'))
def show_section(section):
    for row in index[index.section.eq(section)].itertuples(index=False):
        path = resolve_figure(BUNDLE, DOCUMENT_DIR, row.png, row.png_sha256)
        display(Image(filename=str(path), width=800))
        link(row.data_file, 'Angka pembentuk ' + row.figure_id)
def show_table(relative, limit=24):
    data = pd.read_csv(BUNDLE / relative)
    display(data.head(limit))
    link(relative, 'Unduh tabel lengkap: ' + relative)
print('Policy:', {POLICY!r}, '| sumber:', BUNDLE.name)
'''))
    sections = [
        ('coverage', '1. Coverage penjelasan', 'Denominator bahasa hanya respons berpenjelasan; empat status teks tetap dihitung dari seluruh respons.'),
        ('answer_status', 'Status jawaban', 'Status scoring r5 tidak diubah. Akurasi hanya di tabel per subset bersama commit_rate; deontology/virtue memakai distribusi jawaban.'),
        ('verbosity', 'Panjang respons', 'Median jumlah kata dihitung hanya pada penjelasan eligible. Tokenisasi mengikuti Regional r5, bukan token API.'),
        ('labels_accommodation', '2–3. Label regional dan akomodasi', 'Label, sapaan, adopsi perspektif dan pernyataan tidak relevan adalah kategori terpisah; penyebutan tidak membuktikan endorsement.'),
        ('paired_effects', '4. Efek berpasangan', 'Grafik fokus rujukan budaya umum ditentukan menurut konstruk, bukan p-value. CI numerik adalah CI median delta, bukan CI paired rank-biserial. Semua 56 outcome tersedia pada tabel lengkap. Binary menggunakan delta proporsi; display pp = ×100 sekali. Holm: keluarga 4 versus N, 3 I−E, 24 antaridentitas. BH antar-outcome tersimpan terpisah.'),
        ('general_ethical', '5A. Bahasa umum dan ethical framing', 'Profil ini deskriptif: tidak menempelkan p-value density pada prevalensi. Seluruh 53 marker tersedia pada katalog dan penjelajah di bawah.'),
        ('regional_profile', '5B. Seluruh 23 kategori regional', '23 marker × 17 persona × 6 model = 2.346 sel unik; pengulangan N pada empat identitas menghasilkan 2.760 sel tampilan tanpa menambah observasi.'),
        ('response_cards', '5C. Cara regional diungkapkan', 'Enam triplet purposif, model/subset/soal sama dalam tiap triplet. Pilihan tidak memakai ground truth atau p-value. Cuplikan kontigu maksimal 100 kata; sorotan memakai offset lexical surface, bukan raw final. Label-only tetap ditampilkan sebagai kontras. Lihat HTML untuk teks utuh.'),
        ('diversity', '5D. Keragaman kategori', 'Jumlah kategori berbeda dengan count > 0 (0–23), bukan keragaman semantik/kosakata menyeluruh. Satu span dapat termasuk lebih dari satu kategori; count kategori tidak dideduplikasi menjadi token.'),
        ('diversity_mean', 'Ringkasan keragaman', 'Rerata ini bergantung pada kamus serta panjang respons. Tabel per rentang panjang wajib dibaca bersama.'),
        ('length_diagnostic', 'Diagnosis panjang', 'Tidak ada pembagian jumlah kategori dengan panjang sebagai skor richness baru; strata panjang disajikan deskriptif.'),
        ('context_diagnostic', '5E. Overlap literal dengan konteks', 'Casefold + normalisasi spasi saja. Flag persona dan soal independen; keduanya dapat true. Tidak cocok literal bukan bukti berasal dari model. Prompt user diverifikasi oleh audit korpus. Overlap assistant warmup unavailable: isinya bervariasi antar-request dan 490 request implicit tidak memuat balasan assistant. Snapshot kelompok tidak disubstitusikan untuk konteks per respons.'),
        ('sensitivity', '6. Sensitivitas dan komposisi subset', 'Gabungan berbobot respons, bukan rerata p-value subset. Sensitivitas both-committed: 12 I−E ×56 outcome dan 24 antaridentitas ×21 core; tidak tersedia versus N atau regional-direct. Efek lama–baru berbeda sumber v2/r5 dan eligibility; perubahan bukan otomatis regresi.'),
    ]
    for section, title, description in sections:
        if section in set(index.section):
            cells += [md('## ' + title + '\n\n' + description), code(f'show_section({section!r})')]
    cells += [code("link('evidence/response_cards.html', 'Buka seluruh respons, soal, dan sorotan')\nshow_table('evidence/response_index.csv', 18)"),
              md('## Tabel lengkap dan penjelajah\n\nFilter variabel `marker_id` untuk menelusuri count, density, prevalensi dan efek. Raw count selalu tersedia; NA dibedakan dari nol.'),
              code("show_table('tables/marker_catalog.csv', 53)\nshow_table('tables/coverage.csv')\nshow_table('tables/pairing_exclusions.csv', 20)\nlink('tables/source_provenance_audit.json', 'Audit request dan skenario sumber')\nshow_table('tables/scoring_per_subset.csv')"),
              code("marker_id = 'culture_general_reference'\nprofiles = pd.read_csv(BUNDLE / 'tables/all_marker_profiles.csv')\ndisplay(profiles[profiles.marker_id.eq(marker_id)])\ncontrasts = pd.read_csv(BUNDLE / 'tables/statistics/paired_contrasts.csv.gz')\ndisplay(contrasts[contrasts.outcome_id.eq(marker_id)].head(30))\nlink('tables/statistics/paired_contrasts.csv.gz', 'Seluruh kontras, effect size, CI, p mentah, Holm dan BH')"),
              code("show_table('tables/diversity_by_length.csv')\nshow_table('evidence/named_terms.csv')\nshow_table('evidence/context_overlap.csv')\nshow_table('evidence/prompt_provenance.csv')\nshow_table('tables/statistics/subset_weights.csv')\nshow_table('tables/statistics/old_new_comparison.csv.gz')\nlink('tables/statistics/subset_effect_diagnostics.csv', 'Diagnosis efek per subset — tanpa p-value baru')\nlink('tables/statistics/pair_composition.csv', 'Denominator tiap kontras dan komposisi subset')"),
              md('''## Keterbatasan dan reproduksi

Kamus masih memiliki false positive yang diketahui dan belum menjadi human gold. Kemiripan literal tidak menangkap variasi morfologi, parafrasa, atau rujukan semantik. Mixed-script bukan deteksi bahasa. Perbandingan implicit turut memuat topik/riwayat warmup. Subset deontology tidak menerima excuse; virtue tidak mengikuti konstruk benchmark asli. Analisis bahasa tidak memperbaiki cacat tersebut.

Kelengkapan 23 kategori bukan bukti seluruh cara budaya diungkapkan sudah tertangkap. Notebook/result v2 tetap historis, sedangkan p-value r5 per-subset tidak dipindahkan ke analisis gabungan ini.

Bangun ulang pada run ID baru:

```powershell
python -X utf8 -m src.MAIN_PARSING.ETHICS.build_regional_ethics_presentation --run-id reproduksi_regional_01 --source result/ETHICS/reports/regional_lm_v3_20260916_r5 --stage all
```
''')]
    return nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': dict(name='python3', display_name='Python 3', language='python'),
                                                         'language_info': {'name': 'python'}, 'result_status': 'candidate', 'policy': POLICY})


def execute_notebook(path, cwd):
    from .religion_ethics_gsm8k_style_notebook import execute_notebook as execute
    previous = Path.cwd()
    try:
        os.chdir(cwd)
        notebook = execute(nbformat.read(path, as_version=4))
        nbformat.write(notebook, path)
    finally:
        os.chdir(previous)
    return notebook


def validate_notebook(path, bundle, document_dir=None):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    codes = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if any(cell.execution_count is None or any(o.output_type == 'error' for o in cell.outputs) for cell in codes):
        raise ValueError('Notebook has unexecuted cells or errors')
    pngs = sum('image/png' in o.get('data', {}) for cell in codes for o in cell.outputs)
    if pngs != len(pd.read_csv(Path(bundle) / 'figure_index.csv')):
        raise ValueError('Notebook image count mismatch')
    links = 0
    for cell in codes:
        for output in cell.outputs:
            text = output.get('data', {}).get('text/markdown', '')
            for target in re.findall(r'\]\(<([^>]+)>\)', text):
                target = unquote(target)
                if Path(target).is_absolute() or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target):
                    raise ValueError('Notebook links must be relative')
                resolved = (Path(document_dir or Path(path).parent) / target).resolve()
                if not resolved.is_relative_to(ROOT.resolve()):
                    raise ValueError('Notebook link escapes repository')
                if not resolved.is_file():
                    raise ValueError('Broken notebook link: ' + target)
                links += 1
    return dict(code_cells=len(codes), embedded_pngs=pngs, checked_links=links, notebook_errors=0)
