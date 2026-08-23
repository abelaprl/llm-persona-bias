# Pipeline Linguistic Marker Gender

Status aturan: `gender-lm-v1.2.1-candidate`. Angka dari pipeline ini belum
boleh dipakai sebagai hasil final Bab IV sebelum lembar audit manual selesai.

Untuk dataset ETHICS Gender Persona gabungan tersedia registry terpisah
`ethics-gender-lm-v1.6.0-candidate`. Registry ini memakai permukaan jawaban
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
`result/ETHICS/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0/`.
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

v1.6 memperbaiki hasil audit kontekstual v1.5, terutama penggunaan teknis atau
hipotetis dari `right/wrong`, putusan yang hanya benar di dalam klausa `if`,
validasi concern milik orang ketiga, self-correction generik, serta variasi
refusal dan putusan numerik dengan markup Markdown. Full run v1.6 memakai enam
worker ekstraksi melalui opsi `--extraction-workers 6`; hasil worker paralel
sudah dibandingkan identik dengan ekstraksi sekuensial pada smoke test.

## HASIL AUDIT ETHICS GENDER PERSONA 2026-08-22

Full run v1.6 memproses 81.756 respons unik, 80.840 respons dengan teks yang
dapat dianalisis, 18.521 abstain yang tetap disertakan, 544 respons missing,
1.514 pertanyaan, dan 41 marker. Inferensi tetap dinonaktifkan.

Audit final v1.5 oleh satu reviewer AI memeriksa 3.755 occurrence dan 162
respons recall. Precision mikro adalah 0,9526 dan precision makro 0,9446.
Enam marker berada di bawah 0,90: `affect_empathy_validation` (0,74),
`moral_evaluation_negative` (0,88), `moral_evaluation_positive` (0,71),
`response_stance_committed` (0,72), `response_stance_conditional` (0,81), dan
`stance_self_correction` (0,02). Karena itu v1.5 tidak lolos promosi.

Regresi aturan v1.6 terhadap audit final tersebut menghilangkan 150 dari 178
false positive yang dapat dievaluasi dan mempertahankan 3.362 dari 3.366 true
positive (retention 99,88%). Pada 162 respons recall, regresi v1.6 menghasilkan
0 false negative dan 0 false positive. Ini tetap bukti regresi kandidat, bukan
validasi manusia independen.

Audit bentuk gender terbuka menemukan 50 bentuk berbeda dan semuanya sudah
terdaftar dalam closed lexicon v1.6. `diagnostic_capture_pct` konteks kandidat
tidak ditargetkan 100%: penyebutan `man`, `woman`, atau `gender` yang hanya
menyalin skenario, menyebut pihak ketiga, atau tidak menunjukkan uptake persona
sengaja tetap tersedia untuk audit dan tidak otomatis dihitung sebagai marker.

## HASIL AUDIT ETHICS REGIONAL CULTURAL PERSONA 2026-08-22

Full extraction regional-cultural dijalankan dari file gabungan kanonik terbaru:

```powershell
python -m src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_cultural_dataset_audit `
  --output-dir result/ETHICS/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16 `
  --extraction-workers 1
```

Input `result/ETHICS/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv`
memiliki SHA-256
`09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5`.
Run memproses 154.428 respons unik dari 6 model, 17 kondisi, dan 1.514
pertanyaan; 151.827 surface dapat dianalisis, 40.570 abstain tetap disertakan,
dan 1.823 missing menjadi `NA`. Registry memuat 23 marker regional-cultural dan
30 marker ETHICS umum. Inferensi tetap mati.

Audit AI first-pass beserta seluruh aggregated anchor form dibuat dengan:

```powershell
python -m src.MAIN_PARSING.LINGUISTIC_MARKERS.regional_cultural_codex_adjudication `
  --base-output result/ETHICS/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16 `
  --output-dir result/ETHICS/reports/Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_codex_audit_r1
```

Registry v1.0.16 merupakan hasil pembacaan manual AI atas seluruh 2.017
representative cluster awal dan set occurrence risiko: 768 perluasan
geokultural, 801 hospitality, 583 perspective adoption, 211 direct address,
19 identity irrelevance awal, 90 unmatched-anchor context, serta 17 cluster
identity-irrelevance baru. Exhaustive queue final mencakup 154.428 respons,
204.660 occurrence, dan 1.876 context cluster; seluruh representative cluster
final tercakup catatan audit AI dan open-anchor screen menyisakan 0 potential
miss. Full suite lulus 136/136 test. Hasil tetap candidate sampai dua reviewer
manusia independen menyelesaikan audit dan disagreement resolution; catatan
Codex tidak mengisi kolom reviewer manusia.

## FULL RUN ETHICS RELIGION PERSONA 2026-08-23

Full extraction kandidat awal dijalankan dari sumber kanonik:

`result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv`

SHA-256 sumber adalah
`c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a`,
identik dengan source manifest. Run memproses 154.428 respons unik dari 6
model, 17 skenario, dan 1.514 pertanyaan. Sebanyak 151.022 respons memiliki
permukaan penjelasan yang dapat diukur; 46.060 respons abstain tetap disertakan,
2.381 respons missing menjadi `NA`, dan 1.025 respons berisi teks teknis saja
sehingga juga tidak diberi marker nol.

Bundle kandidat ditulis ke:

`result/ETHICS/reports/Religion_Persona/linguistic_markers_religion_candidate_20260823_v1_0_0/`

Registry `ethics-religion-lm-v1.0.0-candidate` memuat 20 marker agama dan 30
marker ETHICS umum. Keluaran terdiri atas response-level count/presence/density,
ringkasan permukaan, opening, polaritas, prevalence, precision sample 1.081
occurrence, blind recall sample 306 respons, 100 konteks open-anchor yang belum
tertangkap, workbook audit, dan manifest. Inferensi belum dijalankan dan hasil
belum boleh dipromosikan sebelum audit kontekstual serta dua review manusia.

Interpreter `.venv` saat run ini belum memiliki `openpyxl`, sedangkan
`C:\Python314\python.exe` memilikinya. Runner sekarang memeriksa dependency
tersebut sebelum full scan. Perintah yang sudah berhasil untuk melengkapi run
write-once tanpa mengulang ekstraksi adalah:

```powershell
C:\Python314\python.exe -m src.MAIN_PARSING.LINGUISTIC_MARKERS.religion_ethics_dataset_audit `
  --finalize-existing
```

Open-anchor screen masih memuat kandidat recall yang harus diaudit sebelum
v1.0.1, antara lain `insha'Allah`, variasi `halakhah/halachically`, `mitzvahs`,
`Quranic`, serta konteks `commandment(s)`. Bentuk luas seperti `beliefs`,
`sacred`, `holy`, `creator`, dan `fasting` tidak boleh langsung ditambahkan
karena banyak penggunaan nonagama atau medis. Rincian status dan batas
interpretasi ada di
`docs/linguistic_marker_religion_ethics_progress_2026-08-23.md`.

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
