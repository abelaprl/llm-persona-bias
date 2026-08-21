# Pipeline Linguistic Marker Gender

Status aturan: `gender-lm-v1.2.1-candidate`. Angka dari pipeline ini belum
boleh dipakai sebagai hasil final Bab IV sebelum lembar audit manual selesai.

Untuk dataset ETHICS Gender Persona gabungan tersedia registry terpisah
`ethics-gender-lm-v1.5.0-candidate`. Registry ini memakai permukaan jawaban
yang sama, tetapi menambahkan marker wacana moral ETHICS dan marker gender
yang diperlukan oleh dataset tersebut. Kedua registry tidak boleh disamakan
versinya atau dicampur hasil inferensinya.

## SPEC

Tujuan pipeline adalah mengukur perubahan cara enam model utama menyampaikan
jawaban ketika pertanyaan yang sama diberikan pada kondisi `neutral`, satu
persona eksplisit, dan tiga persona implisit untuk identitas laki-laki atau
perempuan. GSM8K dan ETHICS didukung, tetapi tidak pernah digabung menjadi satu
analisis.

Input utama sekarang adalah CSV hasil parsing:

- `result/GMS8K/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/all_models_results_full.csv` untuk GSM8K; serta
- `result/reports/v9_main/{gender,neutral}/<model>/*.csv` untuk ETHICS.

Kolom `question_id` pada CSV GSM8K gabungan tidak dipakai karena sebagian file
berisi nilai konstan. ID pasangan diturunkan dari kolom `file`, misalnya
`gsm8k_00001.json` menjadi `gsm8k_00001`. Raw JSON hanya tersedia sebagai
fallback audit eksplisit melalui `--input-source raw`.

Secara default program memakai seluruh record CSV yang tersedia. Opsi
`--max-records-per-persona` hanya untuk diagnostik dan ditandai sebagai subset
pada manifest.

## DESIGN PLAN

- Hanya `response.content` yang terlihat oleh pengguna yang dianalisis. Field
  `reasoning`/`reasoning_details` tidak pernah menjadi fallback.
- Blok `<think>...</think>` dipisahkan. Tag yang tidak seimbang menghasilkan
  `text_status=unseparable_think` dan semua marker menjadi `NA`.
- Baris protokol jawaban seperti `Answer: 0` dan `Final answer: 18` dikeluarkan
  dari salinan penjelasan dan struktur. `math_boxed_answer` adalah pengecualian
  QC yang memakai visible surface agar kotak pada baris jawaban final tetap
  tercatat tanpa mengubah respons teknis menjadi data linguistik.
- Respons kosong/teknis saja menjadi `NA`, bukan nol. Nol berarti teks tersedia
  dan marker memang tidak ditemukan.
- Setiap marker menyimpan raw count, density per 100 kata, dan presence.
- Tidak ada skor gabungan `reasoning quality`, `total CoT`, atau penjumlahan
  marker yang tumpang tindih.
- Unit inferensi adalah pasangan `question_id` dalam model, benchmark, subset,
  dan identitas yang sama.
- Tujuh planned contrasts dipertahankan; tiga kondisi implisit tidak dirata-rata.
- Outcome numerik memakai Friedman untuk omnibus dan Wilcoxon berpasangan. Sign
  test menjadi hasil utama bila lebih dari separuh delta nol atau sebaran delta
  nonzero sangat asimetris. Effect size adalah paired rank-biserial dan CI 95%
  berasal dari paired bootstrap.
- Density marker dianalisis sebagai outcome numerik utama. Presence seluruh
  marker non-QC juga dibandingkan sebagai analisis pelengkap memakai exact
  McNemar, prevalence difference, matched odds ratio, dan paired bootstrap CI.
- Bootstrap median memakai distribusi order statistic empiris dan bootstrap
  mean memakai multinomial support. Keduanya ekuivalen dengan paired
  non-parametric bootstrap tanpa membuat matriks `resamples x n` yang besar.
- Kategori pembuka memakai Cochran's Q dan exact McNemar per kategori. Holm
  diterapkan sesuai family pengujian yang ditetapkan pada Bab IV.

## IMPLEMENTATION

Jalankan dari root repo:

```powershell
.\.venv\Scripts\python.exe -m src.MAIN_PARSING.LINGUISTIC_MARKERS.run_gender
```

Contoh smoke run GSM8K tanpa statistik penuh:

```powershell
.\.venv\Scripts\python.exe -m src.MAIN_PARSING.LINGUISTIC_MARKERS.run_gender `
  --benchmark gsm8k `
  --models meta-llama_llama-3.1-8b-instruct `
  --max-records-per-persona 20 `
  --extract-only `
  --output-dir result/linguistic_markers/smoke_gender_lm_v1
```

Program menolak output directory yang sudah ada agar hasil lama tidak tertimpa.
Sumber/kondisi yang hilang atau tidak lengkap tidak diubah menjadi respons nol. Pipeline
tetap memproses data yang tersedia, memasukkan gap tersebut ke
`input_inventory.csv`, dan menghasilkan nol pasangan lengkap pada contrast yang
memerlukan kondisi itu.

Full extraction khusus ETHICS Gender Persona dijalankan dengan:

```powershell
.\.venv\Scripts\python.exe -m src.MAIN_PARSING.LINGUISTIC_MARKERS.gender_dataset_audit
```

Input default-nya adalah
`result/ETHICS/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv`
dan output default ditulis ke folder versi baru
`result/ETHICS/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_5_0/`.
Program menolak folder output yang sudah berisi file. Registry gabungan ETHICS
memuat keluarga marker umum, wacana moral, dan gender sebagai outcome terpisah;
tidak ada satu skor gabungan lintas keluarga.

Folder kandidat v1.0, v1.2, v1.3, dan v1.4 tetap dipertahankan sebagai provenance. v1.5
menambahkan pola yang ditemukan lewat audit false-negative v1.2: rujukan
eksplisit `user's gender`, identitas user bersyarat, perspektif gender pada awal
kalimat, konstruksi normatif bergender, neutralization identitas, direct address
parentetik, serta wacana equality/discrimination dan bidang kerja yang didominasi gender.
v1.5 juga menutup false positive v1.4: `societal expectations` atau `societal
norms` tanpa anchor gender tidak lagi dihitung sebagai `gender_role_framing`.

## HASIL SAMPEL YANG TERSEDIA

- `result/linguistic_markers/gender_gsm8k_source_aligned_pilot_v1_2_1_20260817/`
  berisi pilot audit 2.700 respons (50 per model-persona).
- `result/linguistic_markers/gender_gsm8k_full_source_aligned_v1_2_1_20260817/`
  berisi full run GSM8K 71.215 respons dan 10.000 bootstrap resample.

Pilot adalah diagnostic subset. Full run tetap berstatus kandidat sampai audit
precision dan recall dinilai dua penilai. Folder versi v1.0-v1.2 dipertahankan
sebagai provenance dan tidak boleh dicampur dengan v1.2.1.

Jika proses terputus setelah empat file ekstraksi selesai tetapi sebelum
`manifest.json` terbentuk, recovery yang tervalidasi tersedia melalui:

```powershell
python -m src.MAIN_PARSING.LINGUISTIC_MARKERS.resume_gender `
  --output-dir <folder-run-yang-belum-memiliki-manifest>
```

Recovery menolak bundle yang sudah memiliki manifest, beda versi, kurang file,
atau mempunyai key respons duplikat.

## CHECKLIST

- [x] Raw content dan hidden reasoning dipisahkan.
- [x] Missing dibedakan dari zero marker.
- [x] Registry marker memiliki versi, sumber, exclusion rule, dan status validasi.
- [x] Raw count, density, presence, serta surface provenance disimpan.
- [x] Neutral/explicit/implicit 1-3 tidak diagregasi dini.
- [x] Statistik berpasangan per ID soal.
- [x] Effect size, confidence interval, dan Holm correction tersedia.
- [x] Precision audit mengambil occurrence, bukan seluruh hit dari respons
  positif yang dipilih.
- [x] Recall audit mengambil respons lengkap secara buta dari setiap
  model-persona.
- [ ] Audit manual hit marker selesai oleh dua penilai.
- [ ] Audit manual false-negative selesai oleh dua penilai.
- [ ] Inter-rater agreement dihitung.
- [ ] Status registry dipromosikan dari `candidate` menjadi `validated`.

## TESTING

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s src/MAIN_PARSING/LINGUISTIC_MARKERS/tests -v
```

Unit test mencakup positive/negative regex, pemisahan `<think>`, missing value,
baris jawaban teknis, hidden reasoning, pasangan ID soal, effect size, dan Holm.
Untuk pipeline ETHICS, test juga mencakup contextual hedge, distancing,
connective, stance respons, framing moral, polaritas moral, typographic quote,
serta positive/negative context marker gender.

## INTEGRATION

Output baru ditulis ke `result/linguistic_markers/<versi>/`:

| File | Isi |
|---|---|
| `manifest.json` | versi, seed, dependency, cakupan data, registry hash |
| `marker_registry.csv` | definisi dan provenance seluruh marker |
| `input_inventory.csv` | jumlah record CSV dan gap setiap model-persona |
| `response_records.csv` | satu baris per respons dan semua nilai marker |
| `quality_control.csv` | kelengkapan teks per kondisi |
| `audit_marker_hits.csv` | maksimal 50 occurrence per marker untuk audit precision |
| `audit_recall_responses.csv` | respons utuh buta prediksi/correctness untuk audit false-negative |
| `condition_descriptives.csv` | median, IQR, prevalence, raw count |
| `paired_numeric.csv` | tujuh contrast numerik berpasangan |
| `paired_binary.csv` | exact McNemar untuk marker biner |
| `paired_openings.csv` | exact McNemar kategori pembuka |
| `omnibus.csv` | Friedman dan Cochran's Q lima kondisi |

`math_boxed_answer` sengaja diberi peran `quality_control`, bukan bukti
formalitas bahasa. Seluruh marker saat ini tetap kandidat sampai audit manual
selesai; sumber teori tidak otomatis memvalidasi regex operasional.
