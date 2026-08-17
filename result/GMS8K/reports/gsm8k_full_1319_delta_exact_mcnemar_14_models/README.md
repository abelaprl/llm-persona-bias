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
- Kimi dan Trinity dikeluarkan karena hanya dikumpulkan sampai paling banyak 400 soal.
  Model/kondisi lain dapat memiliki sebagian kecil outcome hilang; tabel melaporkan n aktual.
- Exact McNemar tidak diterapkan pada delta agregat karena rata-rata beberapa kondisi bukan
  keluaran biner tunggal.
- Pada visual Avg vs Neutral, `*` berarti sedikitnya satu perbandingan granular penyusun
  terhadap Neutral signifikan setelah Holm (`adjusted p < 0.05`). Bintang bukan hasil uji
  langsung atas batang rerata. Rinciannya tersedia pada
  `gsm8k_aggregate_delta_significance_markers.csv`.
- Visual Imp vs Exp tidak diberi bintang karena analisis pilot ini tidak mendefinisikan uji
  Exact McNemar Implicit-vs-Explicit; p-value persona-vs-Neutral tidak dipindahkan ke batang
  tersebut.

Visual delta adalah `01_delta_avg_vs_neutral_all_available` dan
`02_delta_implicit_vs_explicit_all_available`. Heatmap perubahan akurasi granular adalah
`03_all_conditions_exact_mcnemar_heatmap`. Visual p-value dua panel yang setara dengan
notebook enam model adalah `04_all_conditions_exact_mcnemar_and_holm_pvalue_heatmap`.

Analisis antarpersona yang telah ditetapkan tersedia pada:

- `gsm8k_all_prespecified_paired_effects.csv`: 252 baris (18 perbandingan × 14 model),
  mencakup tujuh primary comparisons untuk Man, tujuh untuk Woman, serta empat matched
  Man-vs-Woman exploratory comparisons;
- `gsm8k_cochran_q_omnibus.csv`: omnibus lima kondisi untuk setiap model dan identitas;
- `05_all_prespecified_persona_effect_size_heatmap.png`: Cohen's h dan keputusan setelah
  Holm/Cochran's Q;
- `06_all_prespecified_persona_exact_mcnemar_holm_pvalue_heatmap.png`: raw Exact McNemar
  dan Holm-adjusted p-value untuk seluruh 18 perbandingan yang ditetapkan.

Koreksi Holm diterapkan secara terpisah dalam family Man-primary (7), Woman-primary (7),
dan matched Man-vs-Woman exploratory (4). Perbandingan silang yang mencampur gender dan
varian framing berbeda, misalnya Man I1-vs-Woman I3, tidak dibuat.
