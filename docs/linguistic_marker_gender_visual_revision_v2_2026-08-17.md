# Handoff Revisi Visual Linguistic Marker Gender v2

Tanggal: 17 Agustus 2026  
Workspace: `D:\TugasAkhir`  
Status: **kode, full computation, 13 figur, dan notebook executed selesai**  
Status ilmiah: **candidate**, karena audit manual precision/recall belum diisi

## 1. Alasan revisi

Visual kandidat v1.2.1 dinilai tidak cukup representatif dan sulit dibaca.
Masalah utamanya bukan sekadar estetika:

1. I1, I2, dan I3 belum menjadi unit visual yang konsisten;
2. signifikansi terlalu dominan dibanding besar efek;
3. tidak ada analisis sensitivitas terhadap correctness;
4. opening category diperlakukan seperti sekumpulan binary outcome independen;
5. coverage belum ditampilkan per model × sembilan persona;
6. normalisasi per 100 kata belum disertai profil word count; dan
7. word cloud tidak memberikan skala yang dapat dibaca.

Artefak v1 tidak dihapus atau ditimpa. Revisi disimpan sebagai versi baru.

## 2. Dasar metodologis yang dibaca ulang

Revisi mengikuti:

- `AGENT.md` pada repo eksperimen;
- `D:\ProposalTA-18222008\AGENT.md`;
- Bab II bagian definisi marker;
- Bab III bagian variabel dan desain berpasangan;
- Bab IV bagian visible final-answer surface, density per 100 kata, planned
  contrast, Wilcoxon/sign test, rank-biserial, bootstrap CI, dan Holm;
- Bab V yang menandai notebook lama sebagai nonfinal;
- Bab VI bagian Gender GSM8K dan definisi konteks I1/I2/I3; serta
- referensi `docs/ref/linguistic marker yang udah aku coba.zip`.

Prinsip yang diambil dari referensi lama adalah penggunaan delta terhadap
baseline, komposisi opening, dan pemisahan model signature dari persona effect.
Angka lama dari referensi tidak dipakai karena sumber tersebut mencakup set
model dan versi ekstraksi yang berbeda. Radar chart dan word cloud lama tidak
dipertahankan karena skala dan nilai eksaknya sulit dibandingkan.

## 3. Implementasi statistik baru

File yang berubah:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/statistics.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_loaders_and_statistics.py`
- `scripts/build_gender_lm_visual_revision_v2.py`
- `scripts/create_gender_lm_visual_notebook_v2.py`
- `scripts/build_gendered_lexicon_analysis.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_gendered_lexicon_analysis.py`

Tambahan utama:

### Correctness sensitivity

Fungsi `build_paired_numeric_correctness_sensitivity` melakukan pairing pada
`model_id × question_id` terlebih dahulu, kemudian mempertahankan pasangan jika
respons kiri dan kanan sama-sama benar. Ini mencegah dua kondisi dibandingkan
menggunakan himpunan soal benar yang berbeda.

Analisis utama tetap seluruh respons (`correctness_filter_applied=false`).
Subset both-correct hanya uji sensitivitas.

### Opening multinomial

Fungsi `build_paired_opening_multinomial` menjalankan Stuart–Maxwell marginal
homogeneity per model, identitas, dan planned contrast. Effect size deskriptif
yang dilaporkan adalah:

- `same_category_rate`; dan
- `marginal_total_variation` antara dua margin opening.

Uji per kategori lama tetap tersedia pada source run untuk audit, tetapi narasi
utama opening menggunakan komposisi dan uji joint ini.

## 4. Output final v2

Source run:

`result/linguistic_markers/gender_gsm8k_full_source_aligned_v1_2_1_20260817/`

Bundle visual baru:

`result/linguistic_markers/gender_gsm8k_bab4_visual_revision_v2_20260817/`

Notebook executed:

`notebooks/linguistic_marker_gender_gsm8k_bab4_visual_revision_v2.ipynb`

Notebook memilih source dan bundle terbaru secara dinamis. Perubahan jumlah
respons memunculkan warning, bukan hard assertion pada 71.215.

### Sepuluh figur

1. `01_coverage_model_persona.png`: exact coverage per model × sembilan persona;
2. `02_verbosity_word_count_median_iqr.png`: median dan IQR word count;
3. `03_numeric_effect_volcano_vs_neutral.png`: E/I1/I2/I3 vs neutral;
4. `04_numeric_effect_volcano_implicit_vs_explicit.png`: I1/I2/I3 vs explicit;
5. `05_proxy_presence_forest_structure_logic.png`: structure, causal, LaTeX;
6. `06_proxy_presence_forest_stance_voice.png`: hedge, self-correction, we;
7. `07_correctness_sensitivity_effect_concordance.png`: all vs both-correct;
8. `08_opening_composition_100pct.png`: opening composition;
9. `09_ranked_lexical_lollipop_vs_neutral.png`: token vs neutral; dan
10. `10_ranked_lexical_lollipop_implicit_vs_explicit.png`: token vs explicit.

Volcano memakai rank-biserial pada sumbu-x dan `−log10(p_holm)` pada sumbu-y.
Ukuran titik mengikuti jumlah pasangan nonzero agar rank-biserial ekstrem dari
marker sangat sparse tidak terlihat memiliki dukungan yang sama dengan efek
yang muncul pada ratusan pasangan.

Forest memakai delta prevalence dalam percentage points dan 95% paired
bootstrap CI. Setiap kolom adalah I1, I2, atau I3; proxy tidak dirata-ratakan.

Lollipop memakai selisih response-level token prevalence, paired by question,
lalu mean setara per model. Token perlu mempunyai arah sama pada minimal empat
dari enam model. Ini tetap analisis eksploratif, bukan uji hipotesis.

### Tambahan cell kata gender eksplisit

Notebook yang sama sekarang memuat tiga figur tambahan; tidak dibuat notebook
baru:

11. `11_gendered_terms_wordcloud_by_persona.png`: kata gender aktual per persona;
12. `12_gender_pronoun_prevalence_by_model_persona.png`: prevalensi respons yang
    memakai pronomina maskulin/feminin per model dan persona; dan
13. `13_gender_pronoun_proxy_forest.png`: delta prevalensi berpasangan I1/I2/I3
    terhadap neutral beserta 95% bootstrap CI.

Word cloud pada figur 11 bukan word cloud token bebas. Vocabulary ditutup
sebelum analisis menjadi 52 kata yang dapat diaudit: `he`, `him`, `his`,
`himself`, `she`, `her`, `hers`, `herself`, serta human nouns seperti `man`,
`woman`, `boy`, `girl`, `father`, `mother`, `husband`, `wife`, dan bentuk
jamaknya. Pencocokan menggunakan batas token eksak, sehingga `he` di dalam
`the` atau `theme` tidak dihitung. Ukuran kata adalah mean setara per model dari
occurrences per 10.000 kata visible-response.

Seluruh 71.215 record pada `response_records.csv` dibaca. Perhitungan kata
hanya memakai 70.248 record dengan `text_available=true` dan field
`visible_text`; prompt, raw JSON, serta hidden reasoning tidak digunakan.
Figur 12 dan 13 disediakan agar pola pada word cloud tetap dapat diperiksa pada
sumbu numerik yang presisi dan dengan desain berpasangan per
`model_id x question_id`.

## 5. Angka audit dan hasil penting

- source responses: 71.215;
- visible text available: 70.248;
- technical-only: 802;
- missing: 165;
- coverage rows model × persona: 54;
- correctness sensitivity planned numeric tests: 1.596;
- paired joint opening tests: 84;
- opening composition rows: 660;
- lexical identity × contrast groups: 14;
- closed gender terms: 52;
- paired gender-pronoun tests: 168;
- bootstrap resamples final: 10.000.

Coverage terendah terdapat pada GPT-OSS 20B man-I1: 1.145 dari target 1.319
(86,81%). Karena itu coverage tidak boleh lagi hanya ditampilkan sebagai total
per model.

Pada kontras terhadap neutral, median pasangan yang bertahan dalam sensitivity
both-correct adalah sekitar 86,4%–87,7%. Di antara outcome primary dengan
minimal 20 perubahan nonzero pada kedua analisis, arah efek sama pada sekitar
89,2%–96,9% kasus. Terdapat satu material sign reversal pada woman I1 vs
neutral, yaitu `stance_assumption` pada Ministral 8B. Namun hasil ini tidak
signifikan setelah Holm pada analisis utama maupun sensitivity (`p_holm=1`),
sehingga dicatat sebagai ketidakstabilan arah, bukan temuan substantif. Rincian
lengkap ada pada `correctness_sensitivity_comparison.csv`.

Opening memperlihatkan heterogenitas yang sangat besar pada DeepSeek R1 32B.
Contoh paling ekstrem adalah man I1 vs neutral dengan marginal total variation
88,29% dan same-category rate 7,52%. Pola ini terlihat langsung pada stacked
composition dan tidak lagi dipecah menjadi sebelas klaim independen.

Temuan marker tetap harus dibaca per model. Contohnya, penurunan inclusive-we
pada DeepSeek R1 untuk I1 vs neutral memiliki dukungan ratusan pasangan,
sedangkan beberapa rank-biserial ±1 pada marker lain berasal dari sangat sedikit
pasangan nonzero. Inilah alasan ukuran titik volcano mengodekan support.

## 6. Validasi

Validasi final:

- 48 unit tests lulus;
- 13 PNG ada;
- word cloud hanya dipakai untuk closed gender lexicon dan selalu didampingi
  grafik prevalensi serta forest plot berskala numerik;
- seluruh 71.215 row CSV dibaca dan 70.248 visible responses dianalisis;
- hidden reasoning tidak digunakan pada analisis leksikon gender;
- seluruh 168 paired gender-pronoun tests mempunyai key unik, `p_holm` valid,
  dan CI memuat point estimate;
- seluruh 84 opening joint tests mempunyai key unik;
- seluruh 1.596 sensitivity tests mempunyai key unik;
- pasangan both-correct tidak pernah melebihi jumlah pasangan utama;
- setiap komposisi opening model × identity × condition berjumlah 100%;
- 11 dari 11 code cells notebook mempunyai execution count;
- notebook tidak mempunyai error output.

Command utama:

```powershell
python scripts\build_gender_lm_visual_revision_v2.py `
  --run-dir result\linguistic_markers\gender_gsm8k_full_source_aligned_v1_2_1_20260817 `
  --output-dir result\linguistic_markers\gender_gsm8k_bab4_visual_revision_v2_20260817

jupyter nbconvert --to notebook --execute --inplace `
  notebooks\linguistic_marker_gender_gsm8k_bab4_visual_revision_v2.ipynb `
  --ExecutePreprocessor.timeout=900
```

## 7. Batas klaim yang masih berlaku

Audit manual masih kosong:

- precision occurrence: 850 sampel, 0 dianotasi;
- recall response: 108 sampel, 0 dianotasi.

Karena itu bundle tetap berstatus **candidate pending manual precision/recall
audit**. Visual v2 layak untuk menyusun struktur subbab dan memilih contoh yang
akan dijelaskan, tetapi angka belum boleh disebut sebagai hasil registry yang
telah tervalidasi manusia.
