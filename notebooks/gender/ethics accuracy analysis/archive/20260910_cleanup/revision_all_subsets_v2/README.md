# ETHICS Gender — Visual Revisi Semua Subset

Bundle ini dibuat tanpa mengubah bundle lama. Sumbernya adalah tabel ETHICS
Gender tervalidasi; tidak ada API call, parsing ulang, atau rescoring.

## Cakupan visual

- `01_ethics_status_composition_by_model`: chart keseluruhan ETHICS; status
  `committed`, `abstain`, dan `missing` dipool dari empat subset.
- `02_ethics_status_by_model_persona_heatmap`: status keseluruhan menurut model
  dan persona.
- `03_ethics_commit_rate_by_persona_subset_heatmap`: versi per subset untuk
  **keempat** subset (`commonsense`, `deontology`, `justice`, `virtue`).
- `04`–`05`: alasan *abstain* keseluruhan.
- `06`–`07`: `accuracy_committed` dan *effective success* hanya untuk
  `commonsense` dan `justice` (construct-valid).
- `08`: `pct_answer_wrong` untuk `deontology` dan `virtue` (distribusi stance,
  bukan akurasi moral).
- `09`: uji independensi status terhadap persona.
- `10`: efek persona terhadap coverage untuk keempat subset.
- `11`–`12`: efek persona terhadap akurasi/effective success untuk
  `commonsense` dan `justice`.
- `13`: efek persona terhadap distribusi *stance wrong* untuk `deontology` dan
  `virtue`.

Semua figure tersedia dalam PNG dan SVG. Akurasi tidak dipool menjadi satu
angka lintas empat subset karena `deontology` dan `virtue` memiliki masalah
construct validity yang terdokumentasi di `docs/plan_parsing_ethics.md`.

## Reproduksi

```powershell
python src/analysis/render_ethics_gender_accuracy_standardized.py `
  --output-dir "notebooks/gender/ethics accuracy analysis/revision_all_subsets_v2"
```
