"""Build and execute a presentation notebook without starting a Jupyter server."""
from pathlib import Path
import json
import copy
import html
import os
import re
from urllib.parse import unquote

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config


def build_notebook(run_id):
    cells = []
    def markdown(text):
        cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    markdown('''# Linguistic Marker Religion Persona pada ETHICS

Notebook ini mengikuti alur dan gaya visual notebook Religion GSM8K: panel per identitas, warna model yang konsisten, grafik efek, dan tabel pembentuk. Islam, Christian, Jewish, dan Atheist dibandingkan pada skenario E, I1, I2, dan I3 secara terpisah; N adalah skenario netral.

Seluruh empat subset ETHICS digabung **hanya untuk penanda linguistik**. Satu respons adalah satu observasi; inferensi memasangkan model × subset × pertanyaan yang sama. Teks yang dianalisis adalah final yang tersedia, termasuk respons abstain berteks. Hasil tetap **candidate**, tanpa validasi precision/recall manusia yang selesai.

Warna: Llama 3.1 biru, Llama 3.3 ungu, Ministral hijau, DeepSeek merah, GPT-OSS jingga, dan Qwen sian. Berlian hitam pada grafik deskriptif adalah median enam model. Grafik inferensial tidak menggabungkan model.''')
    code(f'''from pathlib import Path
import hashlib, html, json, os
import pandas as pd
from IPython.display import Image, Markdown, HTML, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src/benchmarks/ethics').is_dir() and (p / 'result').is_dir())
BUNDLE = ROOT / 'result/ETHICS/reports/religion_ethics_gsm8k_style_{run_id}'
ACTIVE_DIR = ROOT / 'notebooks/religion/linguistic marker/ETHICS'
DOCUMENT_DIR = ROOT / 'notebooks/religion/linguistic marker/ETHICS'
LOCAL_FIGURES = ACTIVE_DIR / 'figures/gsm8k_style_{run_id}'
FIGURES = LOCAL_FIGURES if LOCAL_FIGURES.is_dir() else BUNDLE / 'figures'
manifest = json.loads((BUNDLE / 'manifest.json').read_text(encoding='utf-8'))
figure_index = pd.read_csv(BUNDLE / 'figure_index.csv')
pd.set_option('display.max_colwidth', 100)

def show_section(section):
    for record in figure_index[figure_index.section.eq(section)].itertuples():
        path = FIGURES / Path(record.png).name
        expected = manifest['artifact_hashes'][record.png]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Gambar berbeda dari manifest: ' + str(path))
        display(Image(filename=str(path), width=1600))
        display(Markdown(record.caption))
        links = []
        for target, label in [(path, 'PNG'), (FIGURES / Path(record.svg).name, 'SVG'), (BUNDLE / record.table, 'Tabel lengkap')]:
            relative = os.path.relpath(target, DOCUMENT_DIR).replace('\\\\', '/')
            links.append('<a href="' + html.escape(relative, quote=True) + '">' + label + '</a>')
        display(HTML(' · '.join(links)))

def show_table(filename, columns=None, limit=32):
    table = pd.read_csv(BUNDLE / filename)
    if columns is not None:
        table = table[columns]
    display(table.head(limit).style.format(precision=3, na_rep='NA'))
    relative = os.path.relpath(BUNDLE / filename, DOCUMENT_DIR).replace('\\\\', '/')
    display(HTML(f'<p>Menampilkan {{min(limit, len(table)):,}} dari {{len(table):,}} baris; '
                 f'<a href="{{html.escape(relative, quote=True)}}">CSV lengkap</a>.</p>'))

print('Sumber:', manifest['source_bundle'])
print('Respons:', format(manifest['responses'], ','), '| Marker:', manifest['markers'])
print('Status:', manifest['status'])''')

    sections = [
        ('1. Provenance dan cakupan 17 skenario persona',
         'Angka cakupan memakai seluruh respons per model/persona. Label-only dan final kosong dibedakan dari teks linguistik. '
         'Commit rate, abstain, missing, serta availability teks tersedia berdampingan pada tabel; tidak ada akurasi ETHICS gabungan.',
         "show_section('coverage')\nshow_table('tables/coverage.csv', ['model_id', 'persona', 'total', 'committed', 'commit_rate', 'abstain', 'abstain_rate', 'missing', 'missing_rate', 'linguistic_available', 'coverage_pct', 'label_only', 'empty_final', 'abstain_text'], limit=24)"),
        ('2. Verbosity sebagai dasar normalisasi',
         'Titik adalah median jumlah kata per model dan garis vertikal adalah IQR. Neutral diulang sebagai pembanding '
         'pada setiap panel identitas. Density marker memakai jumlah per 100 kata tiap respons, lalu dirata-ratakan; '
         'respons label-only tidak masuk denominator penjelasan.',
         "show_section('verbosity')\nshow_table('tables/coverage.csv', ['model_id', 'persona', 'linguistic_available', 'word_total', 'median_words', 'q25_words', 'q75_words'], limit=20)"),
        ('3. Penyebutan identitas yang sesuai dengan persona',
         'Panel memakai marker identitas yang sesuai: identitas Islam pada panel Islam, dan seterusnya. '
         'Skenario netral memakai marker yang sama dengan panel pembanding. Penyebutan identitas adalah perilaku bahasa, '
         'bukan bukti keyakinan model atau otomatis bukti akomodasi.',
         "show_section('identity')\nshow_table('tables/aligned_identity.csv', ['model_id', 'persona', 'marker_id', 'linguistic_available', 'raw_count', 'prevalence_text', 'mean_response_density'], limit=24)"),
        ('4. Bentuk akomodasi bahasa agama pada explicit',
         'Seperti acuan GSM8K, setiap panel adalah identitas dan setiap warna adalah model. Marker meliputi '
         'Tuhan/deity, kitab, praktik, konsep, formula, perspektif, otoritas, irrelevance, doa, dan respect. '
         'Nama marker adalah definisi operasional kamus kandidat; scope dan referen yang belum pasti tetap membatasi interpretasinya.',
         "show_section('accommodation')"),
        ('5. Istilah agama yang benar-benar muncul',
         'Lollipop menampilkan 12 span teratas per identitas pada explicit. Sumbernya occurrence yang sudah disimpan '
         'oleh detektor v3. Span yang sama pada posisi dan respons yang sama dideduplikasi lintas marker. '
         'Rata-rata density respons dihitung per model, lalu model diberi bobot setara; count mentah dan denominator '
         'tersedia di tabel. Tampilan memakai ranked terms; tidak menambahkan word cloud.',
         "show_section('terms')\nshow_table('figure_tables/05_religious_terms_ranked_explicit.csv', limit=48)"),
        ('6. Efek marker agama terhadap neutral',
         'Setiap identitas memiliki panel E−N, I1−N, I2−N, dan I3−N. Effect adalah selisih prevalence dalam poin '
         'persentase pada pasangan berteks. Garis menunjukkan CI bootstrap 95% marginal. Titik lebih pekat menandai '
         'p Holm < 0,05; titik kosong berarti CI tidak tersedia. Ketidaktersediaan CI tidak disamakan dengan effect nol.',
         "show_section('religion_effect')"),
        ('7. Marker linguistik umum terhadap neutral',
         'Alur panel mengikuti acuan GSM8K, tetapi sumbu mempertahankan metrik ETHICS v3: delta rata-rata density '
         'respons per 100 kata, bukan rank-biserial. Marker ringkasan ditentukan berdasarkan fungsi bahasa sebelum '
         'render, bukan dipilih berdasarkan signifikansi. Seluruh 59 marker tetap tersedia pada CSV deskriptif, '
         'CSV kontras lengkap, dan atlas historis.',
         "show_section('general_effect')"),
        ('8. Perbandingan langsung antaridentitas',
         'Enam pasangan identitas dibandingkan langsung pada E, I1, I2, dan I3, masing-masing dalam gambar terpisah. '
         'Arah effect mengikuti judul kiri−kanan; perbedaan antarpersona tidak disimpulkan dari dua p-value terhadap neutral. '
         'Effect, CI, Holm, dan jumlah pasangan berasal dari inferensi v3 yang sama.',
         "show_section('direct_identity')"),
        ('8B. Implicit dibanding explicit dalam identitas yang sama',
         'I1−E, I2−E, dan I3−E ditampilkan langsung untuk setiap identitas. Ketiga proxy implicit '
         'dipertahankan terpisah. Seluruh model muncul dengan warna yang sama seperti bagian sebelumnya.',
         "show_section('implicit_vs_explicit')"),
        ('9. Sensitivitas terhadap committed, bobot subset, dan kualitas teks',
         'ETHICS v3 menyediakan sensitivitas pasangan yang keduanya committed, bobot empat subset setara, '
         'dan pengeluaran flag kualitas. Analisis utama tetap mencakup abstain berteks. Titik dekat diagonal '
         'memiliki effect yang serupa pada kedua analisis. Satu titik adalah kontras, sehingga titik-titik '
         'bukan observasi independen; pembalikan arah juga tidak otomatis berarti perubahan temuan signifikan.',
         "show_section('sensitivity')\nshow_table('tables/sensitivity_summary.csv', limit=12)"),
        ('10. Bahasa sikap evaluatif dan batas penilaian',
         'Bagian ini menggantikan komposisi opening pada acuan GSM8K. Marker ETHICS bersifat multi-label: '
         'satu respons dapat memuat sikap tegas, syarat, refusal, dan evaluasi sekaligus. Karena itu persentasenya '
         'tidak dijumlahkan menjadi 100%. Marker sikap tegas berbeda dari status committed milik parser jawaban.',
         "show_section('stance')"),
        ('11. Differential availability',
         'Grafik memakai perbedaan ketersediaan teks setiap persona terhadap neutral pada pertanyaan yang sama. '
         'McNemar dan Holm berasal dari tabel availability v3. Ini perlu dibaca bersama efek bersyarat pada teks, '
         'terutama ketika banyak respons berupa label-only. Perbedaan availability tidak mengukur error detektor.',
         "show_section('availability')\nshow_table('tables/availability.csv', ['model_id', 'persona', 'reference', 'metric', 'n_pairs', 'left_available', 'right_available', 'effect_pp', 'p_holm'], limit=24)"),
        ('12. Dasar pembenaran dan tema moral khusus ETHICS',
         'Tambahan domain ETHICS ditampilkan dengan gaya panel identitas yang sama. Framing kewajiban, konsekuensi, '
         'karakter, agama dan budaya dibedakan dari tema leksikal harm, compassion, fairness, consent/autonomy, '
         'boundaries, responsibility, intention, dan honesty/deception. Kemunculan tema bukan ukuran kebenaran moral '
         'atau endorsement, dan kata umum tidak otomatis menjadi bahasa agama.',
         "show_section('moral')"),
    ]
    for title, description, source in sections:
        markdown('## ' + title + '\n\n' + description)
        code(source)

    markdown('''## 13. Kamus, contoh respons utuh, dan status audit

Kamus mencantumkan definisi, sumber teoretis, eksklusi dan status validasi. Contoh berikut berasal dari paket discovery sebelumnya: paket terarah, dilengkapi putaran terbuka pertama untuk model yang tidak ada di paket terarah. Satu contoh per model dipilih menurut urutan paket, bukan sampel representatif atau gold manusia. Teks lengkap tersedia pada panel yang dapat dibuka. Tidak ada contoh yang dipakai untuk menetapkan benar/salah jawaban.''')
    code('''show_table('tables/registry.csv', ['marker_id', 'label', 'family_id', 'operational_definition', 'theoretical_source', 'validation_status'], limit=59)
examples = json.loads((BUNDLE / 'examples.json').read_text(encoding='utf-8'))
for example in examples:
    heading = ' | '.join(str(example.get(key, '')) for key in ['readback_id', 'model_id', 'persona', 'question_id', 'answer_status'])
    provenance = example['example_source_file'] + ' | indeks ' + str(example['example_source_index'])
    display(HTML('<details><summary>' + html.escape(heading) + '</summary><p>' + html.escape(provenance) + '</p><div style="white-space:pre-wrap;line-height:1.6">'
                 + html.escape(example['final_output']) + '</div></details>'))''')
    markdown('''## 14. Cara memakai hasil dan batas kesimpulan

- Mulai dari coverage dan panjang respons, lalu baca effect beserta jumlah pasangan, CI, Holm, serta sensitivitas.
- Prevalence teks hanya menggambarkan respons berteks. Prevalence output-level dan sensitivitas tetap tersedia pada tabel kontras lengkap.
- Holm mengikuti keluarga pengujian v3; hasil tidak menyatakan satu kontrol error global untuk semua keluarga dan semua model.
- Data tetap candidate. Audit manusia, precision/recall independen, saturation discovery, dan sebagian penentuan scope belum selesai.
- Gaya grafik diselaraskan dengan GSM8K. Konstruk dan metrik kedua benchmark tidak otomatis identik.

Notebook ini menampilkan hasil beku. Menjalankan seluruh sel memuat dan memeriksa gambar serta tabel; tidak memanggil API atau menghitung ulang scoring.''')
    reproduction = ('```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 '
                    'src/benchmarks/ethics/build_religion_ethics_gsm8k_style.py '
                    '--run-id 20260918_reproduction_01\n```')
    code(f'''show_table('figure_index.csv', ['figure_id', 'title', 'section', 'rows'], limit=40)
display(Markdown('**Kontras lengkap:** `' + str(BUNDLE / 'tables/paired_contrasts.csv.gz') + '`'))
display(Markdown('**Reproduksi gambar dan notebook pada run baru:**'))
display(Markdown({reproduction!r}))''')
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'ethics_aggregation': 'overall',
        'result_status': 'candidate', 'presentation_style': 'religion_gsm8k_bab4_visual_v1',
        'source_run': 'religion_lm_v3_candidate_20260916_r4_overall', 'presentation_run': run_id})


def execute_notebook(notebook):
    """Execute real cells with IPython rich display capture in a fresh namespace."""
    InteractiveShell.instance(config=Config({'HistoryManager': {'enabled': False, 'hist_file': ':memory:'}}))
    namespace = {'__name__': '__main__'}
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        execution_count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            exec(compile(cell.source, f'<notebook-cell-{execution_count}>', 'exec'), namespace)
        cell.execution_count = execution_count
        cell.outputs = []
        for name, content in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if content:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=content))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        if captured.stderr:
            raise RuntimeError(f'Unexpected stderr in notebook cell {execution_count}: {captured.stderr}')
    nbformat.validate(notebook)
    return notebook


def rebase_notebook(notebook, old_directory, new_directory, root):
    """Keep Markdown, rich-output links and code links valid after relocation."""
    result = copy.deepcopy(notebook)
    old_directory, new_directory, root = map(Path, (old_directory, new_directory, root))

    def relative_target(target):
        target = html.unescape(target.strip('<>'))
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:|^#', target):
            return target
        base, separator, fragment = target.partition('#')
        path = (old_directory / unquote(base)).resolve()
        return os.path.relpath(path, new_directory).replace('\\', '/') + (separator + fragment if separator else '')

    for cell in result.cells:
        if cell.cell_type == 'markdown':
            cell.source = re.sub(r'\]\(([^)]+)\)', lambda match: '](<' + relative_target(match.group(1)) + '>)', cell.source)
        if cell.cell_type == 'code':
            location = new_directory.relative_to(root).as_posix()
            cell.source = re.sub(r'^DOCUMENT_DIR = ROOT / .*$', 'DOCUMENT_DIR = ROOT / ' + repr(location), cell.source, flags=re.MULTILINE)
            for output in cell.get('outputs', []):
                data = output.get('data', {})
                for mime in ['text/html', 'text/markdown']:
                    if mime not in data:
                        continue
                    if mime == 'text/html':
                        data[mime] = re.sub(r'href="([^"]+)"', lambda match: 'href="' + html.escape(relative_target(match.group(1)), quote=True) + '"', data[mime])
                    else:
                        data[mime] = re.sub(r'\]\(([^)]+)\)', lambda match: '](<' + relative_target(match.group(1)) + '>)', data[mime])
    return result
