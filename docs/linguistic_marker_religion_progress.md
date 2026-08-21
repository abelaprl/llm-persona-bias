# Catatan Lengkap Linguistic Marker Religion Persona — GSM8K

Terakhir diperbarui: 18 Agustus 2026  
Status: full dataset selesai diproses; visual dan notebook selesai dieksekusi; audit kontekstual Codex selesai dan terdokumentasi.

## 1. Ringkasan hasil kerja

Analisis linguistic marker Religion Persona pada GSM8K sudah dibangun ulang untuk enam model, 16 kondisi religion persona, dan satu baseline Neutral bersama. Pipeline final yang dipakai adalah:

`religion-gsm8k-lm-v1.3.0-candidate`

Run final:

`result/GMS8K/reports/Religion_Persona/linguistic_markers/religion_gsm8k_full_source_aligned_v1_3_20260818`

Visual final:

`result/GMS8K/reports/Religion_Persona/linguistic_markers/visual/religion_gsm8k_bab4_visual_v1_3_0_20260818`

Notebook final yang sudah dieksekusi:

`notebooks/linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb`

Audit final:

`result/GMS8K/reports/Religion_Persona/linguistic_markers/religion_gsm8k_full_source_aligned_v1_3_20260818/audit_codex_contextual_review_20260818`

Hasil lama tidak ditimpa. Run v1.1 dan v1.2 sengaja dipertahankan sebagai audit trail karena dari kedua run itulah dua recall miss ditemukan.

## 2. Sumber yang sudah dibaca

### 2.1 Aturan repositori dan tugas akhir

- `AGENT.md` di root repositori.
- `D:/ProposalTA-18222008`, termasuk materi Bab IV yang terkait desain eksperimen dan analisis linguistic marker.
- Struktur dan kode repositori `D:/TugasAkhir` yang berhubungan dengan loader, ekstraktor, statistik, visualisasi, dan hasil GSM8K.
- Notebook gender `notebooks/linguistic_marker_gender_gsm8k_bab4_visual_revision_v2.ipynb` sebagai acuan struktur presentasi yang lebih mudah dibaca.

### 2.2 Referensi ZIP dari pengguna

Arsip berikut sudah diperiksa:

`docs/ref/linguistic marker yang udah aku coba.zip`

Isi pentingnya mencakup notebook dan dokumen berikut:

- `linguistic_marker_religion_persona.ipynb`;
- `Analisis_Komprehensif_GSM8K_Religion.docx`;
- `Analisis_Komprehensif_GSM8K_Religion_v2.docx`;
- `Analisis_Komprehensif_GSM8K_Religion_v3.docx`;
- `Dokumen_Analisis_Riset_Linguistic_Marker.docx`;
- `CoT_linguistic_marker_GSM8K_final.ipynb`;
- `CoT_linguistic_marker_GSM8K_revised.ipynb`;
- notebook discourse marker dan analisis gender/ETHICS lain di dalam arsip.

Versi baru tidak menyalin kelemahan utama notebook lama: analisis tidak dibatasi pada satu model, Neutral tidak dihilangkan, I1/I2/I3 tidak digabung, dan inferensi memakai pasangan model–soal yang sama.

## 3. Provenance data final

### 3.1 Religion persona

Sumber:

`result/GMS8K/reports/Religion_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv`

- 126.624 baris;
- 6 model;
- 16 religion persona;
- 1.319 soal per model–persona;
- tidak ada duplikasi `model × persona × question_id`.

SHA-256:

`38a08efd3ebfe90d611e737d368a6e8d6eef3bc56d20594a82b511cfad606489`

### 3.2 Neutral bersama

Sumber:

`result/GMS8K/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv`

Hanya baris `persona_folder == neutral` yang digunakan.

- 7.907 baris aktual dari target 7.914;
- Llama 3.1 8B memiliki 1.317 baris;
- lima model lain masing-masing memiliki 1.318 baris;
- tujuh gap sumber dipertahankan sebagai missing;
- tidak ada duplikasi `model × question_id`.

SHA-256:

`e0fa7f748c0c9c62196671f6c618dd9923f0c03345f2f522659982ff6dfce508`

### 3.3 Total unit analisis

- Target teoritis: 134.538 respons = 6 model × 17 kondisi × 1.319 soal.
- Respons aktual: 134.531.
- Respons dengan visible text: 132.283.
- Respons tanpa visible text: 2.248; nilainya dipertahankan sebagai NA, bukan marker nol.
- Gap source-level: 7.
- Tidak ada kunci respons duplikat.
- `ground_truth` tidak digunakan oleh ekstraktor.
- Hidden reasoning tidak digunakan.
- File lama `result/reports/all_results_combined.csv` tidak digunakan untuk analisis ini.

## 4. Desain analisis

Identitas yang dianalisis:

1. Islam;
2. Christian;
3. Jewish;
4. atheist.

Kondisi setiap identitas:

- Explicit;
- Implicit 1;
- Implicit 2;
- Implicit 3;
- Neutral sebagai baseline bersama.

I1, I2, dan I3 selalu dipertahankan sebagai proxy terpisah. Identitas agama juga tidak dipool menjadi satu kelompok “religion”.

### 4.1 Kontras within-identity

Tujuh kontras per model, identitas, dan outcome:

- E − N;
- I1 − N;
- I2 − N;
- I3 − N;
- I1 − E;
- I2 − E;
- I3 − E.

### 4.2 Kontras langsung antaridentitas

Enam pasangan identitas diuji pada empat kondisi:

- Islam − Christian;
- Islam − Jewish;
- Islam − atheist;
- Christian − Jewish;
- Christian − atheist;
- Jewish − atheist.

Keluarga Holm untuk klaim langsung berisi 24 perbandingan, yaitu enam pasangan identitas × empat kondisi, per model dan outcome.

### 4.3 Statistik

- Outcome numerik/density: paired rank-biserial effect size dan bootstrap CI 95% dengan 10.000 resample.
- Outcome presence: McNemar exact berpasangan, prevalence delta, bootstrap CI 95%, dan Holm correction.
- Opening category: Stuart–Maxwell marginal homogeneity karena kategori opening bersifat komposisional/multinomial.
- Correctness sensitivity: pasangan dibuat terlebih dahulu, kemudian hanya pasangan yang kedua responsnya benar dipertahankan.
- Differential availability: memakai union question IDs supaya baris hilang tidak lenyap sebelum diuji.
- Unit analisis: satu respons model untuk satu soal.
- Densitas: per 100 kata; raw count tetap disimpan.

## 5. Registry marker agama final

Terdapat 12 marker religion-specific:

1. `religion_identity_islam`;
2. `religion_identity_christian`;
3. `religion_identity_jewish`;
4. `religion_identity_atheist`;
5. `religion_generic_reference`;
6. `religion_deity_invocation`;
7. `religion_scripture_authority`;
8. `religion_practice_community`;
9. `religion_formulaic_expression`;
10. `religion_identity_irrelevance`;
11. `religion_blessing_invocation`;
12. `religion_respect_acknowledgement`.

Aturan konteks penting:

- `Christian` telanjang tidak dihitung karena juga merupakan nama tokoh GSM8K.
- `church`, `Christmas`, dan `fast` tidak dipakai sebagai sinyal tunggal karena dapat berasal dari isi soal.
- `Lent` hanya cocok pada bentuk seperti `Lenten`, `during Lent`, atau konteks observance; verba *lent* tidak cocok.
- Bare `belief/beliefs` tidak dihitung karena belum tentu agama.
- `peace be upon you` generik tidak dihitung tanpa formula agama terdaftar.
- Nama kitab yang juga dapat menjadi nama orang, seperti `John`, `Mark`, `James`, `Ruth`, atau `Job`, baru dihitung jika berbentuk sitasi pasal:ayat.

## 6. Marker baru dari audit recall

### 6.1 Sapaan dan formula agama

Audit recall v1.1 menemukan respons yang jelas memuat bahasa agama tetapi belum tertangkap:

- `Assalamu alaikum` / `As-salamu alaykum`;
- `Wa alaikum salam` / `Wa alaykum as-salam`;
- `Shalom` / `Shabbat Shalom`;
- `L'shana haba'ah b'yerushalayim`;
- `Insha'Allah`;
- `Jazak Allah/Jazakum Allah khayran`;
- `Amen/Ameen`;
- `Chag Sameach`;
- `Mazel Tov`.

Scan full dataset sebelum aturan ditulis menghasilkan:

| Bentuk | Respons | Model yang memuat | Kondisi |
|---|---:|---:|---|
| Assalamu alaikum | 2.790 | 3 | Islam Explicit |
| Shalom | 1.615 | 3 | Jewish Explicit |
| Wa alaikum salam | 1.237 | 3 | Islam Explicit |
| Chag Sameach | 100 | 2 | Jewish Explicit |
| Insha'Allah | 89 | 2 | Islam Explicit |
| Amen/Ameen | 47 | 3 | Christian/Islam Explicit |
| Mazel Tov | 47 | 2 | Jewish Explicit |
| Jazak Allah | 18 | 2 | Islam Explicit |
| L'shana haba'ah | 11 | 1 | Jewish Explicit |

Karena satu respons dapat memuat lebih dari satu formula, total per-bentuk tidak boleh dijumlahkan sebagai jumlah respons unik. Marker final menyala pada 4.574 respons unik.

Kandidat spekulatif yang tidak ditemukan di korpus—`Bismillah`, `Alhamdulillah`, `Mashallah`, `Subhanallah`, dan `Baruch Hashem`—tidak dimasukkan.

### 6.2 Sitasi nama kitab + pasal:ayat

Audit recall v1.2 menemukan kutipan `Ecclesiastes 3:1` yang tidak memuat kata umum `Bible` atau `scripture`, sehingga lolos dari marker lama.

Scan full dataset menemukan:

- 238 respons dengan pola nama kitab + pasal:ayat;
- 129 sudah tertangkap v1.2 melalui kata `Bible`, `biblical`, atau `scripture`;
- 109 belum tertangkap;
- seluruh 109 miss hanya berada pada `christian_explicit`:
  - 14 Llama 3.1 8B;
  - 18 Llama 3.3 70B;
  - 77 Ministral 8B.

Aturan v1.3 menambahkan nama kitab dengan syarat wajib diikuti pola chapter:verse, misalnya `Ecclesiastes 3:1`, `1 Corinthians 13:13`, atau `Proverbs 3:5-6`. Syarat ini mencegah nama tokoh biasa menjadi false positive.

## 7. Riwayat audit recall

Blind recall sample berisi 204 respons: 34 per model dan 12 per setiap kondisi persona. Sampel tidak dipilih berdasarkan marker hit.

| Versi | Respons positif manual | Miss | Keputusan |
|---|---:|---:|---|
| v1.1 | 16 | 9 | Tambah formula/salutation dan lanjut audit |
| v1.2 | 16 | 1 | Tambah named scripture citation dan rerun |
| v1.3 | 16 | 0 | Lolos pada blind sample yang diaudit |

Rincian miss v1.1: delapan respons formula/salutation dan satu respons sitasi kitab. Rincian miss v1.2: satu respons sitasi kitab.

## 8. Hasil audit precision dan recall

Reviewer dicatat secara eksplisit sebagai `Codex contextual review, 2026-08-18`, dengan tipe `AI_assistant_contextual_review`. Ini bukan klaim bahwa audit dilakukan oleh anotator manusia independen.

### 8.1 Precision

- 587 hit dari 12 marker diperiksa.
- True positive: 587.
- False positive: 0.
- Observed precision: 100%.
- Exact Clopper–Pearson CI 95% keseluruhan: 99,37%–100%.

Sebelas marker memiliki 50 sampel acak. Marker `religion_identity_irrelevance` hanya memiliki 37 hit total, sehingga seluruh hitnya diperiksa.

### 8.2 Recall

- Blind sample: 204 respons.
- Respons positif manual: 16.
- Respons positif tertangkap: 16.
- Miss: 0.
- Observed response-level recall: 100%.
- Exact CI 95%: 79,41%–100%.

Pada level label marker manual:

- 25 label marker positif;
- 25 tertangkap;
- observed recall 100%;
- exact CI 95%: 86,28%–100%.

Dua kandidat broad screen tidak dihitung sebagai positif:

- `Grace` adalah nama tokoh GSM8K, bukan konsep agama;
- `personal beliefs` tanpa petunjuk agama sengaja tetap dikecualikan.

CI recall masih lebar karena hanya 16 respons positif di blind sample. Karena itu laporan tesis sebaiknya menulis “tidak ditemukan miss pada sampel audit” dan bukan “recall sempurna untuk seluruh korpus”.

## 9. Hasil deskriptif marker baru pada full dataset

### 9.1 Formula agama pada Explicit

| Identitas | Model | Respons bertanda | Prevalence |
|---|---|---:|---:|
| Islam | Llama 3.1 8B | 1.283/1.319 | 97,27% |
| Islam | Llama 3.3 70B | 1.083/1.319 | 82,11% |
| Islam | Ministral 8B | 471/1.319 | 35,71% |
| Jewish | Llama 3.1 8B | 529/1.319 | 40,11% |
| Jewish | Llama 3.3 70B | 960/1.319 | 72,78% |
| Jewish | Ministral 8B | 242/1.319 | 18,35% |
| Christian | Llama 3.1 8B | 6/1.319 | 0,45% |

DeepSeek R1, GPT-OSS 20B, dan Qwen3 Next tidak menghasilkan formula terdaftar pada kondisi Explicit. Marker ini juga tidak muncul pada Neutral atau ketiga proxy implicit dalam data yang tersedia.

Interpretasi yang aman: persona Explicit memicu accommodation formulaic yang sangat kuat pada sebagian model, bukan perilaku universal semua model.

### 9.2 Scripture authority pada Explicit setelah perbaikan v1.3

| Identitas | Model | Respons bertanda | Prevalence |
|---|---|---:|---:|
| Christian | Llama 3.1 8B | 106/1.319 | 8,04% |
| Christian | Llama 3.3 70B | 63/1.319 | 4,78% |
| Christian | Ministral 8B | 122/1.319 | 9,25% |
| Jewish | Llama 3.1 8B | 5/1.319 | 0,38% |

Total marker scripture authority menyala pada 296 respons. Kenaikan dari v1.2 berasal dari 109 sitasi bernama yang sebelumnya tidak tertangkap.

### 9.3 Implikasi substantif

- Efek religion persona paling jelas pada Explicit.
- Heterogenitas antar-model besar dan harus menjadi bagian utama narasi hasil.
- I1, I2, dan I3 umumnya jauh lebih kecil serta tidak boleh dilebur menjadi satu efek implicit.
- Sapaan/formula agama merupakan outcome penting tersendiri; menggabungkannya ke word cloud saja akan menyembunyikan perbedaan prevalence yang sangat besar.
- Sitasi kitab perlu dipisahkan dari sekadar penyebutan identitas Christian karena keduanya mengukur bentuk accommodation yang berbeda.

## 10. Kode yang dibuat atau diperbarui

Pipeline utama:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_config.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_csv_loaders.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_registry.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_statistics.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/run_religion.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_religion_pipeline.py`.

Visual dan notebook:

- `scripts/build_religion_lm_visual_v1.py`;
- `scripts/create_religion_lm_visual_notebook_v1.py`.

Audit baru:

- `scripts/audit_religion_formula_candidates.py`;
- `scripts/audit_religion_recall_screen.py`;
- `scripts/audit_religion_scripture_citations.py`;
- `scripts/finalize_religion_lm_audit_v1_3.py`.

## 11. Isi run final

Run v1.3 menghasilkan:

- `response_records.csv`;
- `marker_registry.csv`;
- `input_inventory.csv`;
- `quality_control.csv`;
- `audit_marker_hits.csv`;
- `audit_recall_responses.csv`;
- `condition_descriptives.csv`;
- `paired_numeric.csv`;
- `correctness_sensitivity_paired_numeric.csv`;
- `paired_binary.csv`;
- `paired_opening_multinomial.csv`;
- `direct_identity_numeric.csv`;
- `direct_identity_correctness_sensitivity.csv`;
- `direct_identity_binary.csv`;
- `direct_identity_availability.csv`;
- `direct_identity_opening_multinomial.csv`;
- `manifest.json`.

Audit bundle menghasilkan:

- `precision_review.csv`;
- `precision_metrics.csv`;
- `recall_review.csv`;
- `recall_metrics.csv`;
- `audit_iteration_history.csv`;
- `audit_manifest.json`.

## 12. Visual dan notebook

Visual bundle v1.3 berisi 22 PNG dan tabel pendamping. Figur mencakup:

- coverage model × persona;
- verbosity per identitas dan kondisi;
- aligned identity mentions;
- accommodation marker pada Explicit;
- lollipop istilah agama;
- word cloud suplemen berbasis closed lexicon;
- forest plot E/I1/I2/I3 vs Neutral per identitas;
- general linguistic effects vs Neutral;
- direct identity comparisons pada E/I1/I2/I3;
- correctness sensitivity;
- opening composition 100%;
- direct opening comparison;
- differential availability.

Notebook final memiliki:

- 28 cell;
- 14 code cell;
- seluruh 14 code cell sudah dieksekusi;
- 0 execution error;
- 22 PNG tertanam di output cell;
- ukuran sekitar 8,50 MB.

Tulisan defensif seperti “ini bukan uji ...” tidak ditempatkan di dalam gambar. Penjelasan metodologis berada di markdown notebook dan dokumen ini.

## 13. Pengujian

Perintah:

```powershell
python -m unittest discover -s src/MAIN_PARSING/LINGUISTIC_MARKERS/tests -v
```

Hasil akhir:

- 66/66 test lulus;
- termasuk ground-truth isolation, hidden-reasoning exclusion, missing-as-NA, pairing, McNemar, Stuart–Maxwell, correctness sensitivity, direct identity comparison, formulaic expression, dan named scripture citation.

Notebook dieksekusi dengan:

```powershell
jupyter nbconvert --to notebook --execute --inplace `
  notebooks/linguistic_marker_religion_gsm8k_bab4_visual_v1.ipynb `
  --ExecutePreprocessor.timeout=900
```

## 14. Cara rerun dari nol

Gunakan path output baru karena pipeline write-once:

```powershell
python -m src.MAIN_PARSING.LINGUISTIC_MARKERS.run_religion `
  --output-dir result/GMS8K/reports/Religion_Persona/linguistic_markers/<nama_run_baru> `
  --bootstrap-resamples 10000 `
  --audit-per-marker 50 `
  --audit-recall-per-condition 2
```

Bangun visual ke folder baru:

```powershell
python scripts/build_religion_lm_visual_v1.py `
  --run-dir result/GMS8K/reports/Religion_Persona/linguistic_markers/<nama_run_baru> `
  --output-dir result/GMS8K/reports/Religion_Persona/linguistic_markers/visual/<nama_visual_baru>
```

## 15. Tambahan yang disarankan untuk Bab IV

Bab IV sebaiknya menambahkan secara eksplisit:

1. definisi 12 marker religion-specific;
2. alasan `Christian`, `Lent`, `beliefs`, dan nama kitab memerlukan aturan konteks;
3. definisi marker sapaan/formula agama;
4. definisi sitasi kitab sebagai nama kitab + pasal:ayat;
5. audit precision 50 hit per marker atau seluruh hit jika total kurang dari 50;
6. blind recall sample 204 respons yang distratifikasi model dan kondisi;
7. iterasi aturan ketika recall miss ditemukan;
8. exact CI untuk precision dan recall audit;
9. status reviewer sebagai AI contextual review dan kebutuhan konfirmasi manusia independen;
10. family Holm 24 perbandingan untuk direct identity claims;
11. Stuart–Maxwell untuk opening category;
12. pair-then-filter untuk correctness sensitivity;
13. differential availability dari union question IDs;
14. pemisahan hasil GSM8K dari ETHICS, khususnya untuk refusal.

## 16. Batas interpretasi

- Pipeline code tetap berlabel `candidate` karena audit ini dilakukan oleh Codex, bukan anotator manusia independen.
- Tidak ditemukannya miss pada 204 respons tidak membuktikan recall 100% untuk seluruh korpus; CI response-level masih 79,41%–100%.
- Formula agama sangat terkonsentrasi pada tiga model dan kondisi Explicit. Klaim tidak boleh digeneralisasi menjadi “semua model melakukan ini”.
- Missing visible text, terutama pada GPT-OSS, tetap dapat memengaruhi estimasi antarpersona; differential availability harus dibaca bersama hasil marker.
- Word cloud hanya suplemen. Angka yang dikutip di Bab Hasil harus berasal dari prevalence table, lollipop, forest plot, atau CSV inferensial.
- Opening refusal pada analisis Religion GSM8K tidak boleh diasumsikan setara dengan pola refusal pada ETHICS.

## 17. File yang harus dipakai

Untuk analisis dan penulisan berikutnya, gunakan hanya:

- run v1.3;
- visual bundle v1.3.0;
- audit bundle tanggal 18 Agustus 2026;
- notebook Religion yang sudah dieksekusi.

Run v1.1 dan v1.2 hanya untuk riwayat audit, bukan sumber angka final.
