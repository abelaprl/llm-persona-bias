# Dataset

Dataset dikelompokkan berdasarkan benchmark. Nama file asli dipertahankan.

| Folder | Isi | Jumlah file |
|---|---|---:|
| `eli5/` | ELI5 dan sampelnya | 3 |
| `ethics/` | ETHICS per subset, gabungan, dan sampel | 20 |
| `gsm8k/` | GSM8K train dan test | 2 |
| `mmlu_redux/` | MMLU Redux | 1 |
| `squad/` | SQuAD v1/v2, sampel JSON/CSV, dan 2 skrip sampling | 12 |
| `truthfulqa/` | TruthfulQA dan sampelnya | 2 |

Total: 38 file dataset dan 2 skrip sampling. Reorganisasi mencakup semua
file yang sebelumnya berada langsung di `data/`; tidak ada sampling baru.

Jalankan skrip dari akar repositori (`D:\TugasAkhir`). Contoh path:

```text
data/gsm8k/gsm8k_test.json
data/ethics/ETHICS_test_4subsets_10percent_from200.json
data/squad/SQuAD_validation.json
```

Skrip sampling SQuAD berada di `data/squad/gen_samples.py` dan
`data/squad/SQuAD_20_samples.py`. Skrip tersebut menghasilkan ulang sampel
jika dijalankan; pemindahan folder tidak menjalankannya.

Path default runner ETHICS v9 masih menunggu persetujuan sesuai `AGENTS.md`.
Untuk menggunakannya dengan struktur baru, tambahkan argumen berikut ke
perintah runner yang biasa digunakan:

```text
--samples_file data/ethics/ETHICS_test_4subsets_10percent_from200.json
```

Pemetaan lengkap path lama ke path baru beserta SHA-256 tersedia di
[`docs/data_reorganization_20261011.csv`](../docs/data_reorganization_20261011.csv).
Dokumen audit historis, notebook arsip, dan hasil eksperimen dapat masih
menyebut path lama; gunakan pemetaan tersebut ketika menelusuri artefak lama.

Catatan pengerjaan 11 Oktober 2026:

- SPEC: mengelompokkan seluruh 40 file lama berdasarkan dataset.
- DESIGN: enam folder dengan nama file asli; referensi aktif mengikuti folder baru.
- IMPLEMENTATION: 40 file dipindahkan; path di 32 file Python, 7 notebook,
  dan 4 panduan menjalankan skrip diperbarui. Peta repositori di `CLAUDE.md`
  dan aturan pengecualian dataset besar di `.gitignore` ikut disesuaikan.
- CHECKLIST: [x] seluruh file dipindahkan; [x] isi dataset terjaga;
  [x] pemetaan path dan hash dicatat; [ ] path default runner v9 menunggu izin.
- TESTING: 38 hash dataset cocok sebelum/sesudah, 32 file Python lolos
  pemeriksaan sintaks, dan struktur serta output tersimpan 7 notebook identik
  selain sumber sel yang memuat path. Tidak menjalankan ulang eksperimen.
- INTEGRATION: gunakan path per dataset untuk pekerjaan berikutnya; tambahkan
  `--samples_file` seperti contoh di atas saat memakai runner v9.
