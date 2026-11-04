# ETHICS Gender Accuracy Analysis

Notebook ini adalah bundle presentasi ETHICS Gender yang mengikuti aturan visual
notebook Gender GSM8K.

## Sumber

Sumber read-only:

`D:\TugasAkhir\result\ETHICS\reports\Gender_Persona\_combined\persona_publication_analysis_v2_readable_paired_holm_six_models`

Versi sumber: `ethics-gender-publication-paired-holm-v2-readable-20260821`.

Bundle mencakup 81.756 respons dari 6 model, 9 persona, dan 4 subset. Status
sumber: 62.691 `committed`, 18.521 `abstain`, dan 544 `missing`.

## Artefak utama

- `ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb` — notebook yang sudah
  dieksekusi dan memuat tabel, validasi, serta seluruh visual.
- `figures/` — 13 stem visual dalam format PNG dan SVG.
- `tables/` — tabel turunan status, penyebab `abstain`, subset valid, subset
  stance, dan paired effects.
- `bundle_manifest.json` — provenance, jumlah baris, hash sumber, dan daftar
  artefak.
- `output_index.csv` — hash setiap tabel dan visual lokal.

## Kontrak interpretasi

- `accuracy_committed` hanya digunakan untuk `commonsense` dan `justice`.
- `deontology` dan `virtue` ditampilkan sebagai distribusi `stance wrong`, bukan
  akurasi moral.
- `missing` adalah kegagalan infrastruktur dan tidak dipulihkan secara
  linguistik.
- Penyebab `abstain` dilaporkan sebagai `% dari abstain` dan `% dari
  non-missing`.
- Jangan membandingkan akurasi persona tanpa melaporkan `commit_rate` dan
  status jawaban.

## Bangun ulang

Dari root repositori:

```powershell
C:\Python314\python.exe src\analysis\render_ethics_gender_accuracy_standardized.py
C:\Python314\python.exe src\analysis\build_ethics_gender_accuracy_complete_notebook.py
C:\Users\ASUS\AppData\Roaming\Python\Python314\Scripts\jupyter-nbconvert.exe `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  "notebooks\gender\ethics accuracy analysis\ETHICS_Gender_Accuracy_Analysis_Lengkap.ipynb"
```

Perintah di atas hanya menulis ke folder notebook ini; bundle `result\ETHICS`
tidak ditimpa.
