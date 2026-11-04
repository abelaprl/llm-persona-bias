# Arsip utilitas migrasi metrik ETHICS

Batch ini dibuat pada 2026-11-04 untuk memindahkan utilitas migrasi sekali
pakai dari `src/benchmarks/ethics/`. Tidak ada file yang dihapus atau diubah
isinya.

## Dasar keputusan

Kelima skrip memenuhi seluruh kriteria berikut pada saat audit:

- tidak di-import atau direferensikan kode lain;
- tidak direferensikan dokumentasi atau notebook aktif;
- tidak memiliki CLI atau `if __name__ == "__main__"`;
- menjalankan mutasi notebook/CSV langsung saat modul dimuat;
- berasal dari commit migrasi `050a33cd065c3d7e024ef69349fe28a0c467b4c8`.

## Manifest

| File | Path asal | SHA-256 sebelum dipindahkan |
|---|---|---|
| `add_two_metric_images_to_notebooks.py` | `src/benchmarks/ethics/add_two_metric_images_to_notebooks.py` | `3431D2EE599B2F3D41B3023B2CC25C0A78B5847EC93C697EEB60B0EDB300505F` |
| `add_two_metrics_to_existing_tables.py` | `src/benchmarks/ethics/add_two_metrics_to_existing_tables.py` | `69DFFDD831F4E335254D67C41E09D49AFFE1AC4E79A78D0608133F734E9F1FDF` |
| `fix_gender_notebook_metric_text.py` | `src/benchmarks/ethics/fix_gender_notebook_metric_text.py` | `B7421AF9A2421DDB1DF5E8E8823E84594D0C468D411EE20350DDBF5DA5A2D172` |
| `replace_old_metric_docs.py` | `src/benchmarks/ethics/replace_old_metric_docs.py` | `D2DB55B2F4EB5E2581CAF4B258F0278231D62A84264BAC39B3DE66EDA2B60CA7` |
| `update_two_metrics_docs.py` | `src/benchmarks/ethics/update_two_metrics_docs.py` | `A38BEB7C6EDFB62B9ECB28ADC00A0D3330A7EED9C404A832BCA18F547CCA1AD9` |

Untuk memulihkan skrip, pindahkan file yang diperlukan kembali ke path asalnya.
