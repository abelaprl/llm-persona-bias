# Audit linguistic marker ETHICS Gender Persona — Codex R1

Tanggal audit: 2026-08-22
Status: **kandidat; belum divalidasi manusia independen**

## Kesimpulan

Registry v1.5 belum layak dipromosikan. Audit menemukan pemakaian kata yang
secara leksikal cocok tetapi secara fungsi bukan marker yang dimaksud, terutama
komentar teknis tentang format biner, klausa hipotetis, concern milik orang
ketiga, dan self-correction generik. Aturan sudah diperbaiki sebagai
`ethics-gender-lm-v1.6.0-candidate` dan diterapkan ulang ke seluruh dataset.

## Cakupan data

- Input: `result/benchmarks/ethics/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv`
- SHA-256 input: `9a442ffeda5b3308b21fd9c398ab2633e3ad835d41346f5fc1d4ed302b5ee5a4`
- Respons unik: 81.756
- Teks tersedia: 80.840
- Abstain yang tetap disertakan: 18.521
- Missing: 544
- Pertanyaan unik dan skenario tersedia: 1.514/1.514
- Marker: 41, masing-masing memiliki count, density per 100 kata, dan presence
- Inferensi: tidak dijalankan

## Audit final v1.5

- Precision sample: 3.755 occurrence
- True positive: 3.577
- False positive: 178
- Precision mikro: 0,9525965
- Precision makro: 0,9446341
- Recall sample: 162 respons lengkap
- False negative: 34
- Automatic hit yang dibuang reviewer: 5
- Koreksi label kontekstual lintas pass 5–6: 27

Marker di bawah ambang precision 0,90:

| Marker | TP | FP | Precision |
|---|---:|---:|---:|
| `affect_empathy_validation` | 74 | 26 | 0,74 |
| `moral_evaluation_negative` | 88 | 12 | 0,88 |
| `moral_evaluation_positive` | 71 | 29 | 0,71 |
| `response_stance_committed` | 72 | 28 | 0,72 |
| `response_stance_conditional` | 81 | 19 | 0,81 |
| `stance_self_correction` | 1 | 49 | 0,02 |

## Perbaikan v1.6

- Self-correction dipersempit ke koreksi diri yang eksplisit.
- Putusan committed menolak mapping opsi, pertanyaan, dan verdict yang hanya
  berlaku di dalam klausa `if`.
- Conditional stance membedakan kondisi moral dari kondisi teknis, faktual,
  jadwal, model training, dan contoh generik.
- Empathy validation membedakan concern pengguna dari concern orang ketiga.
- Moral polarity menyaring penggunaan tata bahasa, label, dan format biner,
  tetapi mempertahankan evaluasi eksplisit terhadap action/behavior.
- Variasi refusal, numeric verdict, markup Markdown, dan rare-exception frame
  ditambahkan berdasarkan false-negative audit.
- Ekstraksi full dataset dapat memakai proses paralel dengan hasil yang sudah
  dibandingkan identik terhadap jalur sekuensial pada smoke test.

## Regresi v1.6 terhadap audit final

- Baris precision yang dapat dievaluasi: 3.544
- False positive v1.5: 178
- False positive yang dieliminasi v1.6: 150 (84,27%)
- True positive v1.5: 3.366
- True positive yang dipertahankan v1.6: 3.362 (99,88%)
- Precision span yang tetap diprediksi: 3.362 / (3.362 + 28) = 99,17%
- Pada 162 respons recall: 0 false negative dan 0 false positive

Empat true-positive span `affect_empathy_validation` tidak dipertahankan oleh
filter v1.6 yang konservatif. Hal ini tidak menghasilkan false negative pada
audit recall tingkat respons, tetapi tetap harus diperiksa penilai manusia.

## Cakupan kata dan pola gender

Audit bentuk terbuka menemukan 50 bentuk gender berbeda; seluruhnya terdaftar
dalam closed lexicon v1.6 dan tidak ada bentuk open-form yang belum terdaftar.
Pipeline juga menyimpan 8.502 candidate contexts, termasuk konteks yang tidak
menjadi marker. Capture konteks kandidat tidak dimaksa 100%, karena penyebutan
`man`, `woman`, atau `gender` dapat sekadar menyalin skenario, membahas pihak
ketiga, atau tidak menunjukkan persona uptake. Baris seperti itu tersedia di
`gender_unmatched_context_audit.csv` untuk pemeriksaan, bukan otomatis dianggap
false negative.

## Artefak utama

- Audit v1.5 final: `result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_5_0/audit_codex_r1_20260822_v6_final/`
- Regresi v1.6 final: `docs/audit_artifacts_ethics_gender_v16_20260822_v8_final/`
- Full run v1.6: `result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0/`

## Batas klaim

Audit ini dilakukan oleh satu reviewer AI dengan codebook dan pemeriksaan
konteks, bukan dua penilai manusia yang independen dan buta prediksi. Registry
harus tetap berstatus `candidate_pending_manual_audit`. Jangan menjalankan atau
melaporkan inferensi Bab IV sebagai tervalidasi sebelum audit manusia,
penyelesaian disagreement, dan inter-rater agreement selesai.
