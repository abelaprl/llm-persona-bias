# Progres Penanda Linguistik Religion Persona pada ETHICS — 23–24 Agustus 2026

Status: full-corpus extraction, adjudikasi AI R1, dan audit exhaustif AI R2
kandidat v1.1.10 selesai. Registry tetap
`candidate_pending_two_independent_human_reviews`; inferensi statistik belum
dijalankan.

## Scope dan integritas sumber

Unit analisis adalah satu respons model untuk satu pertanyaan. Hanya penjelasan
yang terlihat oleh pengguna yang diukur. Hidden reasoning, ground truth, dan
baris jawaban teknis tidak menjadi permukaan marker.

Input kanonik:

`result/benchmarks/ethics/reports/Religion_Persona/_combined/all_models_results_full.csv`

SHA-256:

`c10a998ce65073e4adbd4a77e69cc1105540d0d600ceff9e8062d375c299533a`

| Ukuran | Nilai |
|---|---:|
| Respons/key unik | 154.428 |
| Duplikasi `model × persona × question_id` | 0 |
| Model | 6 |
| Kondisi/persona | 17 |
| Pertanyaan unik | 1.514 |
| Surface yang dapat diukur | 151.022 |
| Respons abstain yang tetap disertakan | 46.060 |
| Respons missing yang menjadi `NA` | 2.381 |
| Respons teknis saja | 1.025 |
| Marker Religion | 20 |
| Marker ETHICS umum | 30 |

Satu registry diterapkan identik kepada semua model dan kondisi Islam,
Christian, Jewish, atheist, serta neutral. Explicit, implicit 1, implicit 2,
implicit 3, dan neutral tidak digabung lebih awal.

## Registry kandidat v1.1.10

Versi aktif adalah `ethics-religion-lm-v1.1.10-candidate`. Kode utama:

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/ethics_religion_registry.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/extractor.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_ethics_dataset_audit.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_ethics_codex_adjudication.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_ethics_exhaustive_review.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/religion_ethics_exhaustive_adjudication.py`;
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_ethics_religion_registry.py`.

Audit berulang menambahkan bentuk recall yang terbukti pada konteks lengkap,
antara lain Quranic/Qur'anic, `halakhah`, `kavod habrit`, `mitzvahs`/`mitzvaso`,
`insha'Allah`, prophet, commandment/verse berjangkar agama, fasting sebagai
observance, `fear of Heaven`, doa vokatif `Lord, help...` dan `Lord, we ... help
us`, `fire of hell`, `atonement for ... souls`, `for the sake of heaven`, istilah
soul tradisi-spesifik, serta konstruksi divine Creator.

Precision guard memisahkan good/bad faith termasuk bentuk ber-hyphen, faith
sebagai trust kepada sistem/institusi, pencipta karya/aplikasi walaupun
Bible/Torah muncul berdekatan, `sacred ritual` kuliner, lowercase `lent` sebagai
verba, fasting medis, Salvation Army, idiom heaven/hell/soul, disclaimer AI
tanpa soul, `Soul Surfer`, redemption/repurposing sekuler, analogi
commandment/verse, gelar Lord, dan `Lord of the Rings`.

## Validasi korpus dan audit

Seluruh 20 marker Religion dihitung ulang terhadap 154.428 respons. Staging
v1.1.10 memverifikasi ordered key dan surface metric terhadap bundle dasar.
Delta occurrence v1.1.8 → v1.1.10 adalah:

- `religion_generic_reference`: −90;
- `religion_afterlife_soteriology`: +146;
- `religion_scripture_reference`: +252;
- `religion_deity_invocation`: +2 neto; empat ordinary `creator` yang lolos
  karena co-occurrence Bible/Torah dihapus pada v1.1.10.

Kandidat final memiliki 44.607 respons dengan setidaknya satu marker Religion
dan 496.331 total occurrence. Beberapa jumlah respons bertanda:

| Marker | Respons bertanda |
|---|---:|
| `religion_moral_authority_framing` | 20.991 |
| `religion_scripture_reference` | 20.876 |
| `religion_generic_reference` | 17.978 |
| `religion_practice_community` | 16.573 |
| `religion_deity_invocation` | 13.635 |
| `religion_perspective_adoption` | 8.124 |
| `religion_afterlife_soteriology` | 6.269 |
| `religion_formulaic_expression` | 810 |
| `religion_identity_irrelevance` | 499 |
| `religion_blessing_invocation` | 354 |
| `religion_persona_direct_address` | 301 |

Angka tersebut adalah deteksi linguistik deskriptif, bukan efek kausal persona.
Scenario overlap tetap tersedia untuk mencegah atribusi uptake otomatis.

Bundle dan audit write-once:

- kandidat final:
  `result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/`;
- adjudikasi R1:
  `result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r1/`;
- queue exhaustif R2:
  `result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r2_exhaustive/`;
- adjudikasi exhaustif R2 final:
  `result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r2_adjudicated_v3/`.

R1 menghasilkan 1.760 `TP_OPERATIONAL`, 273 `NO_MISS_OBSERVED`, 33
`REVIEWED_INTENTIONAL_EXCLUSION`, dan 200 unmatched-anchor
`INTENTIONAL_EXCLUSION`.

R2 merekonstruksi seluruh 496.331 occurrence pada 7.014 normalized context
cluster dan menyaring 154.428 respons memakai lexical surface yang sama dengan
extractor. Hasil final: 0 potential false-positive occurrence, 0 potential
miss, 5.501 documented local exclusion, dan 3.406 respons tanpa teks yang
ditandai tidak dapat direview. Semua kolom reviewer manusia tetap kosong.

## Pengujian

- Test khusus ETHICS Religion: 38/38 lulus.
- Full suite Linguistic Markers: 174/174 lulus.
- Full-corpus re-extraction: 154.428/154.428 key tervalidasi.
- Response-level gzip memuat 154.428 key unik.

## Validasi lanjutan Codex — 24 Agustus 2026

- Full suite Linguistic Markers diulang di luar sandbox: **174/174 test lulus**
  dalam 44,906 detik. Uji registry Religion sendiri tetap **38/38 lulus**.
- Pemeriksaan sintaks AST terhadap 45 file Python di
  `src/MAIN_PARSING/LINGUISTIC_MARKERS/` menghasilkan **0 error**.
- Manifest kandidat mencatat **154.428 response key unik**, **496.331
  occurrence Religion**, dan **7.014 context cluster** di
  `result/benchmarks/ethics/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10_codex_audit_r2_adjudicated_v3/adjudication_manifest.json`.
- `potential_false_positive_occurrences.csv` pada bundle adjudikasi R2 berisi
  **0 baris data** (hanya header). Ini adalah hasil screen AI, bukan bukti
  validasi manusia.
- Workbook `audit_manual_religion.xlsx` dapat dibuka sebagai XLSX dan memiliki
  7 sheet. Queue recall R2 berisi **154.428 baris**; seluruh nilai
  `codex_second_pass_recall_label`, `human_reviewer_1_label`, dan
  `human_reviewer_2_label` masih kosong.
- Pemeriksaan ini tidak menjalankan inferensi statistik, tidak mengubah input
  kanonik, dan tidak mempromosikan registry. `ground_truth` tetap tidak
  digunakan sebagai teks marker atau label recall.

## Yang masih pending

- Dua reviewer manusia independen menilai precision dan blind recall sample.
- Disagreement resolution dan inter-rater agreement dihitung.
- Registry baru boleh dipromosikan dari `candidate` menjadi `validated` setelah
  dua langkah tersebut selesai.
- Inferensi statistik dan klaim Bab Hasil tetap ditahan sampai promosi.

Dengan batas ini, marker Religion selesai pada tingkat implementasi,
full-corpus validation, dan audit AI untuk semua subset/kondisi ETHICS. Yang
belum selesai adalah validasi manusia independen, bukan coverage pipeline.
