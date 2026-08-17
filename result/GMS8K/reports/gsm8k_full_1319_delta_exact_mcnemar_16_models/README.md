# Analisis GSM8K seluruh soal tersedia

Artefak ini mereproduksi definisi delta pada `Analisa Bias Gender (2).docx` dan mengganti
uji granular two-proportion z-test dengan perbandingan berpasangan yang sesuai.

- Rentang target: question ID 1-1319.
- Delta Avg vs Neutral: mean empat akurasi persona per gender dikurangi akurasi Neutral.
- Delta Imp vs Exp: mean tiga kondisi implicit dikurangi kondisi explicit per gender.
- Perbandingan: delapan kondisi Gender terhadap Neutral pada model yang sama.
- Uji: two-sided Exact McNemar pada pasangan question ID yang tersedia.
- Multiple comparisons: Holm correction dalam delapan perbandingan per model.
- Effect size: paired risk difference (percentage points) dan marginal Cohen's h.
- Kedua sisi setiap pasangan diambil dari keluarga laporan hasil parsing yang sama; raw JSON
  tidak diparsing ulang secara selektif hanya untuk kondisi Neutral.
- Kimi dan Trinity hanya dikumpulkan sampai paling banyak 400 soal. Model/kondisi lain juga
  dapat memiliki sebagian kecil outcome hilang. Setiap tabel melaporkan n aktual.
- Exact McNemar tidak diterapkan pada delta agregat karena rata-rata beberapa kondisi bukan
  keluaran biner tunggal.

Visual delta adalah `01_delta_avg_vs_neutral_all_available` dan
`02_delta_implicit_vs_explicit_all_available`. Heatmap granular Exact McNemar adalah
`03_all_conditions_exact_mcnemar_heatmap`.
