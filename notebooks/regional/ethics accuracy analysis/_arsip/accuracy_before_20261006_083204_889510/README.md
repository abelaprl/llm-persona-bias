# ETHICS Regional-Kultural Accuracy Analysis

**Yang dipakai:** [ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb](ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb).
Seluruh **11 gambar utama** di `figures/` (PNG/SVG) dipakai notebook ini.
`per_subset/` tetap tersedia sebagai detail pendukung, `tables/` sebagai sumber.
Tidak ada gambar Regional yang diarsipkan. [Daftar gambar aktif](active_figures.csv).
[Panduan semua notebook ETHICS](../../ETHICS_MULAI_DI_SINI.md).

Bundle parent ini mengikuti format visual ETHICS Gender/GSM8K: bar chart besar,
anotasi yang terbaca, delta terhadap Neutral, status respons, dan penyebab
abstain.

## Format visual yang diselaraskan

Grafik akurasi memakai palet GSM8K Regional, anotasi 20 pt dengan 3 desimal,
dan sumbu Y 0–1. Heatmap `06`–`09` memakai empat panel berisi 14/14/12/12
perbandingan tanpa memisahkan kelompok persona. Pada file p-value, `part1`
berisi seluruh Exact McNemar dan `part2` seluruh Holm. Nilai dan cakupan tetap.
Reproduksi kedua notebook: `python src/MAIN_PARSING/ETHICS/harmonize_regional_religion_ethics_visuals.py`.

## Cakupan parent

- Status `committed`, `abstain`, dan `missing` dipool dari seluruh empat subset.
- Akurasi parent hanya memakai subset construct-valid `commonsense` dan
  `justice`; `deontology` dan `virtue` tidak dilebur sebagai akurasi moral.
- Empat kategori persona ditampilkan sebagai Explicit dan rata-rata tiga kondisi Implicit,
  ditambah Neutral.
- Rincian empat subset berada sebagai child di `per_subset/`.

## Provenance

Sumber statistik: bundle v3 tervalidasi pada `result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined`.
CSV respons sumber hanya dibaca untuk membentuk tabel status dan alasan abstain;
tidak ada parsing ulang, rescoring, API call, atau perubahan pada `result/`.

## Bangun ulang

```powershell
C:\Python314\python.exe src/analysis/render_ethics_regional_overall_standardized.py
C:\Python314\python.exe src/analysis/build_ethics_regional_combined_accuracy_notebook.py
C:\Users\ASUS\AppData\Roaming\Python\Python314\Scripts\jupyter-nbconvert.exe `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  "notebooks\regional\ethics accuracy analysis\ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb"
```
