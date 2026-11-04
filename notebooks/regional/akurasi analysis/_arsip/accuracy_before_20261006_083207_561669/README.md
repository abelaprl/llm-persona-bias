# GSM8K Regional Cultural Accuracy Analysis

Folder ini adalah artefak additive untuk analisis akurasi persona Regional-Kultural.
Output lama di `result/` dan notebook linguistic-marker di folder lain tidak ditimpa.

## Provenance

- Dokumen Bab VI: `D:\ProposalTA-18222008\Bab VI - Evaluasi.tex`.
- Bab VI Figure 12: `12_regional_absolute_accuracy_explicit_implicit.png`.
- Bab VI Figure 13: `13_regional_delta_implicit_vs_explicit.png`.
- Bab VI Figure 27: `27_regional_all_52_effect_holm_stacked_26_26.png`.
- Tabel aktif: `result/benchmarks/gsm8k/reports/persona_publication_analysis_v5_paired_significance_six_models_bar3_compact_smalllabels_heatmap3_sigbold/tables`.
- Renderer lokal: `src/analysis/render_gsm8k_regional_accuracy_standardized.py`.
- Builder notebook: `src/analysis/build_gsm8k_regional_accuracy_complete_notebook.py`.
- Notebook linguistic-marker GSM8K yang menjadi sumber Gambar 09/11 berada di
  `notebooks/regional cultural/`, bukan folder akurasi ini. Gambar 14 Bab VI
  dirender oleh `D:\ProposalTA-18222008\scripts\refresh_regional_gsm8k_volcano_id.py`.

Renderer akurasi membaca tabel v5 secara read-only. Ia tidak membaca respons mentah,
tidak melakukan parsing/scoring, dan tidak menghitung ulang Exact McNemar, Cochran's Q,
atau Holm.

## Aturan visual

- PNG dan SVG selalu dibuat berpasangan dengan stem yang sama.
- Latar `#FCFBF8`; judul utama dan judul panel bold.
- Label sumbu, legenda, dan angka batang normal; angka signifikansi heatmap bold hanya
  jika `significant_holm_0_05 = true`.
- Figure 12 memakai angka akurasi 17 pt, normal, vertikal.
- Figure 13 memakai angka delta 16 pt, normal, vertikal, dan diletakkan di luar batang.
- Figure 27 memakai judul 23 pt bold, judul panel 19 pt bold, sel non-signifikan normal,
  dan tanpa footnote bawah di dalam gambar. Definisi tanda tetap dijelaskan di notebook
  dan caption Bab VI.

## Batas interpretasi

Bar Regional menggabungkan tiga kondisi implicit sebagai rerata deskriptif; uji utama
tetap membaca I1, I2, dan I3 secara terpisah. Garis Neutral pada Figure 12 adalah
baseline visual dari Neutral Gender karena pasangan Neutral Regional dengan protokol
yang sama tidak tersedia. Jangan memperlakukannya sebagai hasil Exact McNemar
Regional-vs-Neutral.

## Rebuild

Renderer menolak folder output yang sudah berisi file agar artefak versioned lama tidak
tertindih. Artefak saat ini sudah berada di `figures/`. Untuk membuat versi baru, gunakan
folder output baru dan arahkan `FIGURE_DIR` pada builder ke folder tersebut sebelum
membangun notebook:

```powershell
python src/analysis/render_gsm8k_regional_accuracy_standardized.py `
  --output-dir "notebooks/regional/akurasi analysis/figures_v2"
# lalu ubah FIGURE_DIR pada builder ke figures_v2
python src/analysis/build_gsm8k_regional_accuracy_complete_notebook.py
```
