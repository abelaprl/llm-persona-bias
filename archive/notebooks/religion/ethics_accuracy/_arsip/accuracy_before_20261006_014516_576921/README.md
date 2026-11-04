# ETHICS Religion Accuracy Analysis

**Yang dipakai:** [ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb](ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb).
`figures/` berisi **7 gambar utama** (PNG/SVG), `per_subset/` berisi 8 gambar detail.
Gambar yang tidak dipakai berada di `archive/20260910_cleanup/`.
Data di `tables/` tetap diperlukan. [Daftar gambar aktif](active_figures.csv).
[Panduan semua notebook ETHICS](../../ETHICS_MULAI_DI_SINI.md).

## Tampilan gabungan empat panel

Warna batang akurasi dan delta mengikuti palet GSM8K Regional, dipetakan secara
konsisten ke identitas dan skenario; Neutral abu-abu. Grafik akurasi memakai
3 desimal, anotasi 20 pt, serta sumbu Y 0–1. Cakupan data tidak berubah.
Reproduksi bersama Regional: `python src/MAIN_PARSING/ETHICS/harmonize_regional_religion_ethics_visuals.py`.

Grafik `00_ethics_religion_combined_absolute_accuracy` menampilkan kecocokan label
empat subset: jumlah benar / jumlah committed, bukan satu ukuran kemampuan moral.
Penggabungan disetujui pengguna; Implicit Avg berbobot jumlah committed I1–I3.
Coverage dan denominator ditampilkan berdampingan di notebook yang sama.
Reproduksi: `python src/MAIN_PARSING/ETHICS/add_religion_ethics_accuracy_chart.py`.

Notebook yang sama kini menampilkan tiga heatmap gabungan empat subset:
`00_ethics_religion_combined_inference_effect_holm`,
`00_ethics_religion_combined_exact_mcnemar_four_panels`, dan
`00_ethics_religion_combined_holm_pvalue_four_panels` (PNG/SVG).
Masing-masing berisi 52 perbandingan × 6 model dengan empat panel vertikal:
Ateis/Kristen, Islam/Yahudi, Ateis versus agama lain, dan pasangan lainnya.
Seluruh E/I1/I2/I3 untuk satu kelompok tetap berada dalam panel yang sama.

Angka tebal dan `*` selalu berarti p-value Holm < 0,05. Tanda Cohen's h
ditampilkan sebagai A−B sesuai label; kolom gabungan lama memakai B−A sehingga
tanda dibalik hanya saat visualisasi. P-value dan family Holm sumber tidak
diubah. Statistik gabungan memakai kecocokan label empat subset; keterbatasan
konstruk Deontology/Virtue tetap berlaku, sehingga tidak ditafsirkan sebagai
satu ukuran kemampuan moral. Notebook menjelaskan scope dan family Holm.

Untuk memperbarui figure beserta output notebook tanpa menghitung ulang statistik:

```powershell
python src/MAIN_PARSING/ETHICS/update_religion_ethics_existing.py --layout-only
```

Perintah ini memeriksa 312 sel per figure, keutuhan kelompok persona, kecocokan
tanda efek, hash tabel numerik sebelum/sesudah, dan eksekusi notebook.
Bagian berikut menjelaskan bundle parent lama (termasuk scope Commonsense/Justice).

Bundle parent ini mengikuti format visual ETHICS Gender/GSM8K: bar chart besar,
anotasi yang terbaca, delta terhadap Neutral, status respons, dan penyebab
abstain.

## Cakupan parent

- Status `committed`, `abstain`, dan `missing` dipool dari seluruh empat subset.
- Akurasi parent hanya memakai subset construct-valid `commonsense` dan
  `justice`; `deontology` dan `virtue` tidak dilebur sebagai akurasi moral.
- Empat kategori persona ditampilkan sebagai Explicit dan rata-rata tiga kondisi Implicit,
  ditambah Neutral.
- Rincian empat subset berada sebagai child di `per_subset/`.

## Provenance

Sumber statistik: bundle v3 tervalidasi pada `result/benchmarks/ethics/reports/Religion_Persona/_combined`.
CSV respons sumber hanya dibaca untuk membentuk tabel status dan alasan abstain;
tidak ada parsing ulang, rescoring, API call, atau perubahan pada `result/`.

## Bangun ulang

```powershell
C:\Python314\python.exe src/analysis/render_ethics_religion_overall_standardized.py
C:\Python314\python.exe src/analysis/build_ethics_religion_combined_accuracy_notebook.py
C:\Users\ASUS\AppData\Roaming\Python\Python314\Scripts\jupyter-nbconvert.exe `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  "notebooks\religion\ethics accuracy analysis\ETHICS_Religion_Accuracy_Analysis_Lengkap.ipynb"
```
