# Rekonstruksi pilot pemilihan model GSM8K

Artefak ini mengganti grafik two-proportion z-test dalam `Analisa Bias Gender (2).docx`.

- Unit analisis: question ID 1-300 (`gsm8k_00001.json` sampai `gsm8k_00300.json`).
- Perbandingan: delapan kondisi Gender terhadap Neutral pada model yang sama.
- Uji: two-sided Exact McNemar pada pasangan question ID yang tersedia.
- Multiple comparisons: Holm correction dalam delapan perbandingan per model.
- Effect size: paired risk difference (percentage points) dan marginal Cohen's h.
- Kedua sisi setiap pasangan diambil dari keluarga laporan hasil parsing yang sama; raw JSON
  tidak diparsing ulang secara selektif hanya untuk kondisi Neutral.
- OLMo 3.1 32B memiliki 299 outcome Neutral; Kimi memiliki 257 outcome pada
  `man_implicit_3`. Setiap tabel melaporkan `n_pairs` aktual.
- Disebut *rekonstruksi pilot* karena ringkasan pilot 300 soal asli tidak ditemukan dalam audit
  dokumen lama. Sumber outcome dan aturan rekonstruksi dipertahankan dalam tabel provenance.

Visual utama adalah `01_pilot_model_selection_sensitivity_summary`; heatmap lengkap
`02_pilot_all_conditions_exact_mcnemar_heatmap` lebih cocok untuk lampiran.
