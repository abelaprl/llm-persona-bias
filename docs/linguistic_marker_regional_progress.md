# Catatan Pengerjaan Linguistic Marker Regional/Cultural GSM8K

Tanggal pengerjaan: 18 Agustus 2026  
Status: pipeline full-dataset, audit, visualisasi, dan notebook selesai dijalankan.

## 1. Tujuan

Pekerjaan ini membuat analisis linguistic marker untuk persona regional-kultural GSM8K dengan mutu dan struktur yang setara dengan analisis gender dan religion. Empat identitas yang dianalisis adalah:

- Amerika Utara (`north_american`)
- Asia Timur (`east_asian`)
- Asia Selatan (`south_asian`)
- Afrika Sub-Sahara (`sub_saharan_african`)

Kondisi persona tidak digabung:

- Neutral (`N`)
- Explicit (`E`)
- Implicit 1 (`I1`): nama dan nilai sosial
- Implicit 2 (`I2`): perayaan
- Implicit 3 (`I3`): tradisi pernikahan

## 2. Dokumen dan referensi yang dibaca

Analisis diselaraskan dengan:

- `D:\ProposalTA-18222008\Bab IV - Perancangan.tex`
- `D:\ProposalTA-18222008\Bab VI - Evaluasi.tex`
- `D:\TugasAkhir\docs\ref\linguistic marker yang udah aku coba.zip`
- pipeline dan notebook gender yang sudah ada
- pipeline dan notebook religion yang sudah ada

Dokumen dalam ZIP yang dibaca dan dipakai sebagai referensi antara lain:

- `Analisis_Komprehensif_GSM8K_Regional.docx`
- `Dokumen_Analisis_Riset_Linguistic_Marker.docx`
- `Analisis_Linguistic_Marker_Lengkap.docx`
- `Analisis_Linguistic_Marker_GSM8K.docx`

Prinsip Bab IV yang dipertahankan:

- hanya teks jawaban yang terlihat yang dianalisis;
- hidden reasoning dan warmup tidak diperlakukan sebagai jawaban GSM8K;
- count, density per 100 kata, dan presence dibedakan;
- I1, I2, dan I3 adalah konstruk terpisah;
- perbandingan memakai model dan `question_id` yang sama;
- label kawasan dipisahkan dari bahasa budaya yang lebih luas.

## 3. Penggabungan full dataset

Seluruh sumber Regional/Cultural disatukan ke:

`result/GMS8K/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv`

Ringkasan:

- 126.624 baris regional;
- 6 model;
- 16 persona regional;
- 1.319 soal per model-persona;
- 126.624 key model-persona-soal unik;
- tidak ada duplikat key;
- 126.450 baris memiliki `response_content_full`;
- 174 baris tidak memiliki `response_content_full`;
- SHA-256: `f76ab6897fd8c202ab332d9f130a52886edbfc30cd6a4b842d81bea7146fc593`.

Neutral tidak dibuat ulang atau difabrikasi. Sumber Neutral bersama diambil dari source-aligned combined Gender Persona:

`result/GMS8K/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv`

Setelah Neutral ditambahkan, full run berisi:

- target: 134.538 respons;
- aktual: 134.531 respons;
- gap sumber: 7 respons.

File pembangun combined:

`scripts/build_gsm8k_regional_combined_full.py`

Manifest sumber:

- `_combined/all_models_results_full_source_manifest.csv`
- `_combined/all_models_results_full_manifest.json`

## 4. Eksplorasi seluruh korpus sebelum registry

Seluruh 126.624 respons regional dibaca untuk menemukan bentuk bahasa yang benar-benar muncul. Beberapa hasil discovery:

- label lengkap `East Asia`: 718 respons;
- label lengkap `South Asia`: 697 respons;
- label lengkap `Sub-Saharan Africa`: 491 respons;
- label lengkap `North America/North American`: 7 respons;
- `Namaste`: 256 respons, seluruhnya Asia Selatan Explicit;
- direct regional address seperti `Nice to meet you from ...`: lebih dari 1.200 respons;
- `Hello from Sub-Saharan Africa`: 182 respons;
- `Diwali`: 4 kemunculan pada warmup I2 Asia Selatan di discovery awal;
- `mehendi`: 13 kemunculan pada I3 Asia Selatan;
- `red envelope/hongbao`: 14 respons pada I2 Asia Timur;
- `Thanksgiving`: 3 kemunculan pada I2 Amerika Utara;
- `harvest celebration`: 6 respons pada I2 Afrika Sub-Sahara;
- nama lengkap warmup yang terbawa: 2 respons, keduanya `Wei Chen`.

Istilah berikut sengaja tidak dipakai sebagai marker mandiri karena bertabrakan dengan isi soal GSM8K:

- `American`, `USA`, `Chinese`, `China`, `Indian`, `India`, dan `African`;
- `family`, `community`, `values`, dan `background`;
- `custom/customs`, karena banyak merujuk pada petugas bea cukai;
- `collective`, karena dapat berarti total gabungan dalam aritmetika.

## 5. Registry marker regional-kultural v1.1

Pipeline final memakai 32 marker:

- 18 marker umum yang sama dengan pipeline gender/religion;
- 14 marker regional-kultural.

Marker regional-kultural:

1. `culture_region_north_american`
2. `culture_region_east_asian`
3. `culture_region_south_asian`
4. `culture_region_sub_saharan_african`
5. `culture_general_reference`
6. `culture_regional_address`
7. `culture_identity_mirroring`
8. `culture_geographic_greeting`
9. `culture_formulaic_greeting`
10. `culture_named_celebration`
11. `culture_named_tradition`
12. `culture_warmup_name_echo`
13. `culture_meta_framing`
14. `culture_identity_irrelevance`

Versi registry:

`regional-visible-lexicon-v1.1.0-candidate`

Registry tetap berstatus candidate karena konfirmasi manusia independen masih disarankan sebelum klaim final tesis.

## 6. Perbaikan berdasarkan blind recall

Registry awal v1.0 belum menangkap sapaan seperti:

- `Hello from the other side of the world`
- `Hello from across the globe`

Blind sample menemukan satu respons positif yang terlewat. Setelah itu ditambahkan marker:

`culture_geographic_greeting`

Pencarian full-corpus menemukan 140 respons:

- 64 `other side of the world`;
- 76 `across the world/globe`;
- 0 false hit pada `Hello from the AI assistant`.

Distribusi 140 respons tersebut:

- Llama 3.1 8B, Afrika Sub-Sahara Explicit: 7;
- Llama 3.3 70B, Asia Timur Explicit: 5;
- Llama 3.3 70B, Asia Selatan Explicit: 36;
- Llama 3.3 70B, Afrika Sub-Sahara Explicit: 92.

Seluruh pipeline, statistik, visual, dan notebook kemudian dijalankan ulang sebagai v1.1. Hasil v1.0 tidak dicampur dengan hasil v1.1.

## 7. Pipeline dan file kode

File baru atau yang disiapkan untuk Regional/Cultural:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/regional_config.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/regional_registry.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/regional_csv_loaders.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/regional_statistics.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/run_regional.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_regional_pipeline.py`
- `scripts/build_regional_lm_visual_v1.py`
- `scripts/finalize_regional_lm_audit_v1_1.py`
- `scripts/create_regional_lm_visual_notebook_v1.py`

Versi pipeline final:

`regional-gsm8k-lm-v1.1.0-candidate`

Pipeline menghitung:

- descriptives per kondisi;
- paired numeric effects;
- paired binary presence dengan McNemar exact;
- bootstrap CI 95%;
- rank-biserial untuk outcome numerik;
- Holm correction;
- correctness sensitivity setelah pairing;
- opening multinomial dengan Stuart–Maxwell;
- enam perbandingan langsung antarkawasan pada E, I1, I2, dan I3;
- differential text availability.

## 8. Full run final

Direktori run final:

`result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/regional_gsm8k_full_source_aligned_v1_1_20260818`

Parameter:

- full dataset;
- 134.531 respons aktual;
- 32 marker;
- bootstrap 10.000 resample;
- audit precision maksimum 50 hit per marker;
- blind recall 2 respons per model-persona.

File statistik yang dihasilkan:

- `response_records.csv`
- `marker_registry.csv`
- `input_inventory.csv`
- `quality_control.csv`
- `audit_marker_hits.csv`
- `audit_recall_responses.csv`
- `condition_descriptives.csv` — 4.080 baris
- `paired_numeric.csv` — 5.544 baris
- `correctness_sensitivity_paired_numeric.csv` — 5.544 baris
- `paired_binary.csv` — 5.208 baris
- `paired_opening_multinomial.csv` — 168 baris
- `direct_identity_numeric.csv` — 4.752 baris
- `direct_identity_correctness_sensitivity.csv` — 4.752 baris
- `direct_identity_binary.csv` — 4.464 baris
- `direct_identity_availability.csv` — 288 baris
- `direct_identity_opening_multinomial.csv` — 144 baris
- `manifest.json`

## 9. Audit kontekstual

Direktori audit:

`result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/regional_gsm8k_full_source_aligned_v1_1_20260818/audit_codex_contextual_review_20260818`

Precision:

- 480 sampled hit regional ditinjau;
- 480 true positive;
- 0 false positive;
- precision sampel 100%;
- Clopper–Pearson CI 95%: 99,23%–100%.

Blind recall:

- 204 respons acak terstratifikasi;
- 3 respons manual-positive;
- 5 label marker manual-positive;
- 3/3 respons tertangkap setelah revisi v1.1;
- 5/5 label tertangkap;
- CI tetap lebar karena jumlah respons positif hanya tiga.

Interpretasi yang benar: tidak ada miss tambahan yang ditemukan dalam sampel ini. Hasil tersebut bukan bukti bahwa recall seluruh korpus pasti 100%.

File audit:

- `precision_review.csv`
- `precision_metrics.csv`
- `recall_review.csv`
- `recall_metrics.csv`
- `audit_iteration_history.csv`
- `audit_manifest.json`

Reviewer dicatat sebagai AI contextual review, bukan human annotation.

## 10. Visual bundle

Direktori visual final:

`result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/visual/regional_gsm8k_bab4_visual_v1_1_0_20260818`

Jumlah PNG: 22.

Visual mencakup:

- coverage model × 17 persona;
- verbosity per kawasan dan kondisi;
- aligned regional identity mentions;
- regional-cultural accommodation pada Explicit;
- ranked term lollipop;
- word cloud closed lexicon;
- empat forest plot marker budaya vs Neutral;
- empat forest plot marker umum vs Neutral;
- empat forest plot direct comparison untuk E/I1/I2/I3;
- correctness sensitivity;
- opening composition 100%;
- direct opening differences;
- differential availability.

Word cloud hanya memakai istilah regional-budaya yang telah ditentukan. Word cloud tidak dibuat dari semua kata paling sering dalam respons. Lollipop tetap menjadi tampilan kuantitatif utama karena memiliki sumbu dan nilai eksak.

## 11. Notebook final

Notebook:

`notebooks/linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1.ipynb`

Notebook sudah dieksekusi penuh:

- 28 cell;
- 14 code cell;
- execution count 1–14 lengkap;
- 57 output;
- 22 gambar tertanam;
- 0 traceback;
- ukuran sekitar 9 MB.

Isi notebook:

1. provenance dan coverage;
2. verbosity;
3. aligned identity mention;
4. accommodation language;
5. ranked regional terms dan word cloud;
6. marker budaya vs Neutral;
7. marker linguistik umum vs Neutral;
8. direct comparison antarkawasan;
9. correctness sensitivity;
10. opening composition;
11. differential availability;
12. audit dan contoh konteks;
13. implikasi untuk Bab IV dan Bab Hasil.

## 12. Hasil substantif awal

Penyebutan label kawasan pada Explicit sangat heterogen antarmodel.

Prevalence tertinggi:

- Llama 3.1 8B — Asia Timur: 50,57%;
- Llama 3.1 8B — Asia Selatan: 44,35%;
- Llama 3.1 8B — Afrika Sub-Sahara: 30,33%;
- Llama 3.3 70B — Asia Selatan: 8,27%;
- Llama 3.3 70B — Afrika Sub-Sahara: 6,90%;
- Llama 3.3 70B — Asia Timur: 3,87%.

DeepSeek R1 32B dan GPT-OSS 20B tidak menunjukkan penyebutan label kawasan lengkap pada Explicit. Amerika Utara juga hampir tidak disebut oleh seluruh model. Karena itu, hasil tidak boleh diringkas sebagai satu rerata global tanpa memperlihatkan heterogenitas model.

Sapaan budaya/geografis:

- `Namaste` muncul pada Asia Selatan Explicit;
- Llama 3.1 8B: 60 respons atau 4,55%;
- Llama 3.3 70B: 196 respons atau 14,87%;
- sapaan geografis umum paling tinggi pada Llama 3.3 70B Afrika Sub-Sahara Explicit: 92 respons atau 6,98%.

Carryover implicit tetap spesifik terhadap proxy:

- I2 Asia Timur membawa `red envelope/hongbao`;
- I2 Asia Selatan membawa `Diwali/diyas`;
- I2 Amerika Utara membawa `Thanksgiving`;
- I2 Afrika Sub-Sahara membawa `harvest celebration`;
- I3 Asia Selatan membawa `mehendi`;
- I1 hanya menunjukkan dua echo nama lengkap `Wei Chen`.

Correctness sensitivity menunjukkan arah yang relatif stabil:

- persona-vs-Neutral: sekitar 92,9%–98,5% hasil yang didukung pada kedua subset mempertahankan arah;
- direct antarkawasan: sekitar 91,1%–93,0% mempertahankan arah;
- median pair retention sekitar 85,8%–89,8%.

## 13. Missingness dan differential availability

Kehilangan teks terutama terjadi pada GPT-OSS 20B dan kondisi implicit. Contoh gap text availability terbesar pada direct comparison mencapai sekitar 10–11 poin persentase.

Contoh:

- Amerika Utara vs Asia Timur, I1: −10,92 pp;
- Asia Timur vs Asia Selatan, I2: −10,61 pp;
- Asia Timur vs Afrika Sub-Sahara, I2: −10,39 pp.

Karena missingness berbeda menurut kondisi dan kawasan pada GPT-OSS, data hilang tidak boleh langsung dianggap *missing completely at random*. Grafik dan tabel availability disediakan agar caveat ini terlihat.

## 14. Saran penambahan Bab IV dan Bab Hasil

Bab IV sebaiknya menambahkan secara eksplisit:

- definisi empat kawasan dan tiga proxy implicit;
- definisi unit pasangan model×soal;
- enam direct contrasts antarkawasan untuk setiap kondisi;
- Holm correction sebagai family 24 perbandingan per model-marker;
- pemakaian label kawasan lengkap dan aturan eksklusi demonym/country term;
- count, density, dan presence sebagai ukuran yang berbeda;
- opening sebagai outcome multinomial dan Stuart–Maxwell sebagai uji berpasangan;
- correctness sensitivity setelah pairing;
- pemeriksaan differential availability;
- audit precision dan blind recall beserta riwayat revisi registry.

Bab Hasil sebaiknya menekankan:

- heterogenitas antarmodel;
- Explicit jauh lebih kuat daripada implicit pada penyebutan identitas;
- I1, I2, dan I3 tidak boleh diagregasi;
- direct regional comparisons diperlukan untuk klaim perbedaan kawasan;
- lexical carryover implicit sesuai isi masing-masing warmup;
- GPT-OSS memerlukan caveat missingness khusus.

## 15. Validasi kode

Seluruh test suite linguistic marker dijalankan:

```text
Ran 75 tests
OK
```

Tes mencakup extractor umum, surface handling, pairing, McNemar, Stuart–Maxwell, bootstrap, correctness sensitivity, gender, religion, dan regional.

## 16. Perintah reproduksi

Membangun combined source:

```powershell
python scripts/build_gsm8k_regional_combined_full.py
```

Menjalankan full pipeline:

```powershell
python -m src.MAIN_PARSING.LINGUISTIC_MARKERS.run_regional `
  --output-dir result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/regional_gsm8k_full_source_aligned_v1_1_20260818 `
  --bootstrap-resamples 10000 `
  --audit-per-marker 50 `
  --audit-recall-per-condition 2
```

Membangun visual:

```powershell
python scripts/build_regional_lm_visual_v1.py `
  --run-dir result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/regional_gsm8k_full_source_aligned_v1_1_20260818 `
  --output-dir result/GMS8K/reports/Regional_Cultural_Persona/linguistic_markers/visual/regional_gsm8k_bab4_visual_v1_1_0_20260818
```

Menyelesaikan audit:

```powershell
python scripts/finalize_regional_lm_audit_v1_1.py
```

Membuat dan mengeksekusi notebook:

```powershell
python scripts/create_regional_lm_visual_notebook_v1.py
jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  notebooks/linguistic_marker_regional_cultural_gsm8k_bab4_visual_v1.ipynb
```

Menjalankan semua tes:

```powershell
python -m unittest discover -s src/MAIN_PARSING/LINGUISTIC_MARKERS/tests -v
```

## 17. Keterbatasan yang tetap harus disebutkan

- Registry regional masih berstatus candidate.
- Audit precision dilakukan oleh AI contextual review, belum annotator manusia independen.
- Blind recall hanya 204 respons dan hanya tiga respons positif; CI recall lebar.
- Beberapa marker muncul sangat jarang, sehingga absence tidak selalu berarti model tidak sensitif pada persona.
- Respons dapat mengulang isi warmup; marker menunjukkan surface carryover, bukan mekanisme kausal internal model.
- Heterogenitas model sangat besar, sehingga pooled average dapat menyesatkan.
- Missingness GPT-OSS berbeda antarpersona dan perlu disebutkan dalam interpretasi.
