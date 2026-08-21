# Handoff Percakapan dan Implementasi Linguistic Marker Gender

> **Pembaruan direct gender v3:** klaim perbedaan gender sekarang diuji
> langsung sebagai Man minus Woman pada E/I1/I2/I3. Lihat
> `docs/linguistic_marker_gender_direct_v3_2026-08-17.md` dan
> `notebooks/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb`.

> **Pembaruan visual v2:** visual kandidat awal pada dokumen ini sudah
> disupersede oleh revisi effect-first, proxy-separated, correctness-sensitive,
> dan compositional-opening. Lihat
> `docs/linguistic_marker_gender_visual_revision_v2_2026-08-17.md` serta
> `notebooks/linguistic_marker_gender_gsm8k_bab4_visual_revision_v2.ipynb`.
> Notebook v2 tersebut tetap menjadi satu-satunya notebook utama. Cell terbaru
> menganalisis closed gender lexicon (`he/she`, `him/her`, dan human gender
> nouns) dari seluruh CSV serta menampilkan figur 11--13.

Tanggal: 17 Agustus 2026  
Workspace: `D:\TugasAkhir`  
Status kode: **implementasi dan full run selesai**  
Status ilmiah registry: **`gender-lm-v1.2.1-candidate`**, menunggu audit manual

## 1. Tujuan pengguna

Pengguna meminta:

1. membaca referensi linguistic marker lama, `AGENTS.md`, proposal TA, dan repo;
2. membuat linguistic marker yang lebih baik, dimulai dari Gender;
3. memakai CSV yang sudah tersedia, bukan raw JSON sebagai input utama;
4. memastikan versi setiap model dan sumber akurasinya benar;
5. memakai sumber gabungan GSM8K yang source-aligned;
6. menghasilkan sampel yang mudah diperiksa; dan
7. melanjutkan sampai full extraction dan statistik yang dapat dipakai untuk
   menyiapkan Bab IV.

## 2. Sumber data yang dipakai

Input utama Gender GSM8K:

`result/GMS8K/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv`

CSV tersebut dibangun dari source manifest publikasi v5, memakai path dan
SHA-256 sumber yang ditetapkan di:

`result/GMS8K/reports/persona_publication_analysis_v5_paired_significance_six_models/tables/source_manifest.csv`

Builder sumber:

`scripts/build_gsm8k_source_aligned_combined.py`

Keputusan penting:

- `result/reports/all_results_combined.csv` **tidak dipakai** sebagai sumber
  linguistic marker karena response text lima model banyak yang kosong dan key
  kondisi tidak stabil.
- `is_correct` dipertahankan hanya sebagai metadata QC; tidak pernah dipakai
  untuk memilih atau membuang respons.
- Hidden reasoning tidak pernah menjadi fallback visible response.

Dokumentasi audit sumber:

`docs/gsm8k_gender_religion_source_aligned_2026-08-17.md`

## 3. Kode final Gender

Package:

`src/MAIN_PARSING/LINGUISTIC_MARKERS/`

File utama:

- `config.py`: versi, model, persona, input source-aligned, planned contrasts;
- `registry.py`: registry marker v1.2.1, pola, definisi, exclusion rule;
- `surface.py`: visible/lexical/structure surface, `<think>` separation, NA;
- `extractor.py`: count, density per 100 kata, presence, opening category;
- `csv_loaders.py`: CSV-first loader dan stable `question_id` dari kolom file;
- `statistics.py`: descriptives, paired numeric, exact McNemar presence,
  opening, omnibus, Holm, effect size, paired bootstrap;
- `audit_sampler.py`: precision occurrence sampling dan recall response sampling;
- `run_gender.py`: CLI run baru yang menolak overwrite;
- `resume_gender.py`: recovery ketat jika ekstraksi selesai tetapi manifest belum
  terbentuk;
- `tests/`: regression test surface, regex, loader, pairing, dan statistics;
- `README.md`: spesifikasi dan cara menjalankan pipeline.

Builder ringkasan kandidat Bab IV:

`scripts/build_gender_lm_candidate_summary.py`

Builder contrastive word clouds:

`scripts/build_gender_contrastive_wordclouds.py`

## 4. Marker registry v1.2.1

Marker yang diukur:

- reasoning organization: explicit step, bullet, section header;
- epistemic stance: hedge, booster, assumption, self-correction;
- logical connective: causal, contrastive, conditional;
- self-reference: first-person singular dan inclusive we;
- engagement: reader directive;
- textual elaboration: code gloss;
- mathematical presentation: LaTeX display dan plain arithmetic;
- QC/supplementary: boxed answer dan AI identity statement;
- surface outcomes: word count, average sentence length, TTR-300, opening type.

Setiap marker menyimpan:

- raw count;
- density per 100 kata; dan
- presence 0/1.

Tidak ada skor komposit “reasoning quality” atau “total CoT”.

## 5. Revisi penting berdasarkan audit

Perubahan dari v1.1 ke v1.2.1:

- bold sentence seperti `**Josh made a profit ...**` tidak lagi otomatis
  dianggap section header;
- boxed LaTeX bertingkat dideteksi pada visible final-answer surface;
- `probably`, `likely`, `I think/believe/suspect`, dan subject-bounded
  `it/this/that seems/appears` masuk hedge;
- `indeed` dan `in fact` masuk booster;
- `must be` dikeluarkan dari booster karena audit menunjukkan dominasi makna
  deontic/constraint, bukan certainty;
- imperative `Assume ...` masuk assumption;
- `Let's re-evaluate/recalculate/...` masuk self-correction;
- response-initial `Wait/Actually` tidak dihitung self-correction karena belum
  ada visible reasoning yang direvisi;
- inclusive `we will/should/want` ditambahkan;
- `remember that` hanya diterima pada posisi reader-directed imperative, bukan
  `I remember that`;
- plain arithmetic menangkap seluruh multi-operand expression, currency,
  decimal, percent, dan tidak boleh menyeberangi newline;
- precision audit mengambil occurrence secara stratified tanpa mengekspansi
  seluruh population hit di memori;
- recall audit mengambil dua respons lengkap per model-persona dan membutakan
  automatic marker serta correctness.

Perbandingan total v1.1 ke v1.2.1 pada key data yang sama:

| Marker | Count v1.1 | Count v1.2.1 | Present v1.1 | Present v1.2.1 |
|---|---:|---:|---:|---:|
| stance hedge | 15.360 | 19.367 | 1.023 | 1.431 |
| stance booster | 770 | 273 | 603 | 227 |
| stance assumption | 938 | 947 | 636 | 643 |
| stance self-correction | 7.430 | 7.740 | 905 | 1.015 |
| inclusive we | 81.496 | 82.104 | 40.369 | 40.433 |
| reader directive | 60 | 61 | 60 | 61 |
| inline arithmetic | 77.654 | 99.126 | 33.061 | 39.421 |

## 6. Sampel dan full output final

Pilot audit final, 2.700 respons:

`result/linguistic_markers/gender_gsm8k_source_aligned_pilot_v1_2_1_20260817/`

Full result final, 71.215 respons:

`result/linguistic_markers/gender_gsm8k_full_source_aligned_v1_2_1_20260817/`

Ringkasan tabel kandidat Bab IV:

`result/linguistic_markers/gender_gsm8k_bab4_candidate_summary_v1_2_1_20260817/`

Ringkasan tersebut mempertahankan seluruh planned test di source run. File
bernama `holm_significant` hanyalah view `p_holm < 0.05` dan tidak menggantikan
tabel paired lengkap.

Isi full result:

- `response_records.csv`: 71.215 key unik, 94 kolom;
- `marker_registry.csv`;
- `input_inventory.csv`;
- `quality_control.csv`;
- `audit_marker_hits.csv`: 850 occurrence, 50 untuk masing-masing 17 marker
  yang muncul;
- `audit_recall_responses.csv`: 108 respons, dua per 54 model-persona;
- `condition_descriptives.csv`: 1.200 baris;
- `paired_numeric.csv`: 1.596 planned tests;
- `paired_binary.csv`: 1.428 marker-presence tests;
- `paired_openings.csv`: 924 tests;
- `omnibus.csv`: 372 tests;
- `manifest.json`: provenance, version, seed, dependency, dan flags.

Text status full run:

- available: 70.248;
- technical-only: 802;
- missing: 165.

Missing dan technical-only tidak diubah menjadi respons nol linguistik.

### Notebook dan visualisasi

Notebook eksplorasi hasil:

`notebooks/linguistic_marker_gender_gsm8k_v1_2_1.ipynb`

Notebook tersebut sudah dieksekusi penuh: 10 dari 10 code cell selesai tanpa
error output. Output grafik tersimpan di:

`result/linguistic_markers/gender_gsm8k_bab4_candidate_summary_v1_2_1_20260817/images/`

- `01_text_coverage_by_model.png`: available, technical-only, dan missing per
  model;
- `02_explicit_vs_neutral_marker_presence_heatmap.png`: selisih proporsi
  kehadiran marker antara persona gender eksplisit dan neutral;
- `03_primary_numeric_significance_count_heatmap.png`: jumlah planned contrast
  signifikan per model dan outcome; angka ini bukan ukuran besar efek;
- `04_explicit_vs_neutral_opening_heatmap.png`: perubahan tipe pembukaan respons
  antara persona gender eksplisit dan neutral;
- `05_contrastive_wordcloud_personas_vs_neutral.png`: E/I1/I2/I3 dibandingkan
  dengan neutral untuk man dan woman;
- `06_contrastive_wordcloud_implicit_vs_explicit.png`: I1/I2/I3 dibandingkan
  dengan explicit untuk man dan woman.

Word cloud memakai prevalence token biner per respons, dipasangkan berdasarkan
model dan `question_id`. Selisih dihitung per model lalu dirata-ratakan dengan
bobot setara. Token visual harus memiliki arah yang sama pada sedikitnya empat
dari enam model. Frekuensi mentah, prompt, hidden reasoning, correctness, token
LaTeX, dan boilerplate inti GSM8K tidak dipakai sebagai bobot visual. Nilai yang
dapat diaudit tersimpan di `lexical_contrast_token_weights.csv`, sedangkan
parameter lengkapnya ada di `lexical_contrast_metadata.json`.

Interpretasi visual tetap berstatus kandidat Bab IV sampai audit manual
precision dan recall selesai.

## 7. Validasi yang sudah selesai

- 71.215 response key unik;
- semua row memakai `gender-lm-v1.2.1-candidate`;
- `correctness_filter_applied=false`;
- 10.000 bootstrap resample;
- tidak ada duplicate key pada empat tabel inferensial;
- tidak ada p-value `NA` atau di luar rentang 0–1;
- math audit tidak memiliki match lintas newline;
- self-correction audit tidak memiliki match pada offset 0;
- booster audit tidak memiliki `must be`;
- recall audit tidak mengandung kolom correctness atau automatic marker;
- 44 unit tests lulus pada 17 Agustus 2026.

Command test:

```powershell
python -m unittest `
  src.MAIN_PARSING.LINGUISTIC_MARKERS.tests.test_surface_and_markers `
  src.MAIN_PARSING.LINGUISTIC_MARKERS.tests.test_loaders_and_statistics
```

## 8. Status “kode sudah beres?”

**Ya untuk implementasi, full extraction, dan statistik Gender GSM8K.**

**Belum final secara ilmiah** karena pekerjaan manusia berikut masih wajib:

1. dua penilai mengisi `audit_marker_hits.csv` untuk precision;
2. dua penilai mengisi `audit_recall_responses.csv` untuk false-negative;
3. hitung inter-rater agreement;
4. putuskan apakah registry dipromosikan dari `candidate` menjadi `validated`;
5. setelah itu baru angka dirangkai menjadi narasi final Bab IV.

Full tables saat ini boleh dipakai untuk menyiapkan struktur tabel/grafik Bab
IV, tetapi harus diberi label **hasil kandidat**.

## 9. Folder lama dan provenance

Jangan mencampur output berikut dengan final v1.2.1:

- `gender_csv_sample_v1/`;
- `gender_csv_stats_sample_v1/`;
- `gender_csv_sample_v1_incomplete_stats/`;
- `gender_gsm8k_source_aligned_sample_v1/`;
- pilot/full v1.1;
- pilot v1.2.0.

Folder lama sengaja tidak dihapus agar provenance dan perbandingan versi tetap
tersedia.

## 10. Langkah berikut yang disarankan

1. Buka pilot v1.2.1 untuk sanity check pengguna.
2. Siapkan codebook singkat untuk dua penilai.
3. Isi precision dan recall audit secara independen.
4. Hitung agreement dan revisi rule hanya jika audit memerlukan.
5. Kunci versi validated.
6. Bangun tabel/grafik Bab IV dari full paired tables tanpa memilih hasil
   berdasarkan correctness atau arah efek.
