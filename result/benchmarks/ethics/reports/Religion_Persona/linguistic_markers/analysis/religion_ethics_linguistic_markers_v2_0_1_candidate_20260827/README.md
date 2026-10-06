# ETHICS Religion Linguistic Marker v2 — kandidat

Status: **belum thesis-final**. Registry marker masih menunggu dua reviewer
manusia independen. Audit AI exhaustive tidak menggantikan validasi manusia.

## Isi

- `dataset/`: dataset response-level analisis dalam CSV gzip dan Parquet;
- `tables/`: scoring, descriptives, paired inference, omnibus, opening,
  sensitivity, pair counts, dan gerbang validasi;
- `figures/`: setiap visual dalam PNG 300 dpi, PDF, dan SVG;
- `notebooks/`: salinan notebook source, executed notebook, dan HTML;
- `provenance/`: snapshot marker registry;
- `thesis/`: snippet integrasi LaTeX dan catatan batas interpretasi.

Kontras utama adalah I1/I2/I3 minus E (Holm dalam keluarga 3). Kontras terhadap
N adalah sensitivity (Holm dalam keluarga 4). Direct identity memakai enam
pasangan identity × empat kondisi (Holm dalam keluarga 24). Bootstrap memakai
10,000 resample per baris hasil.

`missing` dan `technical_only` dipertahankan sebagai NA. Semua respons visible,
termasuk abstain, masuk analisis utama. Akurasi hanya dipakai sebagai konstruk
utama pada Commonsense dan Justice; Deontology/Virtue memakai distribusi stance.

Sebanyak 84 baris pasangan (`40` within dan `44` direct) memiliki `n_pairs=0`,
seluruhnya khusus outcome `ttr_300` pada GPT-OSS 20B. Baris itu sengaja tetap
ada dengan CI `NA`; tidak ada imputasi atau penghapusan diam-diam. Rinciannya
tersimpan pada `tables/pair_availability_audit.csv`.
