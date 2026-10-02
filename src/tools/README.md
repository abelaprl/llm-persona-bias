# Panduan tools

Folder `src/tools` memuat builder, audit, dan utilitas eksperimen. Nama versi
lama tidak berarti kode tidak dipakai: sebagian entry point dipertahankan agar
eksperimen lama tetap dapat dijalankan.

## Implementasi reusable

Implementasi reusable untuk visualisasi gender berada di
[`src/pipelines/analysis/gender_visuals/`](../pipelines/analysis/gender_visuals/):

- `contrastive_wordclouds.py`
- `lexicon_analysis.py`
- `revision_v2.py`
- `revision_v3.py`

Empat `build_*.py` terkait di folder ini hanya menjadi entry point kompatibilitas
dan meneruskan eksekusi ke package tersebut. Perintah lama tetap valid, tetapi
kode baru sebaiknya mengimpor dari package `src`.

## Visualisasi linguistic marker GSM8K

| Skrip | Peran |
|---|---|
| [build_gender_lm_visual_revision_v3.py](build_gender_lm_visual_revision_v3.py) | Builder visual Gender dengan perbandingan langsung Man–Woman |
| [build_gender_lm_visual_revision_v2.py](build_gender_lm_visual_revision_v2.py) | Builder dan fungsi visual dasar yang juga diimpor v3 |
| [build_gendered_lexicon_analysis.py](build_gendered_lexicon_analysis.py) | Analisis kata berpenanda gender; diimpor builder v2 dan v3 |
| [build_religion_lm_visual_v1.py](build_religion_lm_visual_v1.py) | Builder visual marker Religion |
| [build_regional_lm_visual_v1.py](build_regional_lm_visual_v1.py) | Builder visual marker Regional/Cultural |
| [add_gsm8k_religion_regional_volcano.py](add_gsm8k_religion_regional_volcano.py) | Menambahkan volcano dari statistik tersimpan ke bundle dan notebook baru |
| [render_gsm8k_marker_readable_v1.py](render_gsm8k_marker_readable_v1.py) | Membaca tabel titik volcano dan menghasilkan empat gambar dalam PNG/PDF |

Builder visual yang masih tinggal di folder ini mengimpor package
`src.core.linguistic_markers` secara langsung dari
`src/core/linguistic_markers/`; tidak diperlukan alias package tambahan.

Generator volcano dan perender keterbacaan mempunyai tugas berbeda:
generator menyiapkan tabel titik beserta notebook; perender membaca tabel
tersebut untuk tata letak yang lebih mudah dibaca. Keduanya dipertahankan.

## Perapian 15 September 2026

Dua fungsi visual lama yang tidak dipanggil oleh `main()` beserta lima
konstantanya dibuang dari `render_gsm8k_marker_readable_v1.py`.
Fungsi-fungsi pada alur aktif tetap sama. Tidak ada penghapusan file tambahan.

Tiga belas path di folder ini sudah berstatus penghapusan staged sebelum
cleanup. Daftar dan bukti pemeriksaan tersedia pada
[laporan cleanup scripts](../../docs/cleanup_scripts_20260915_232211/README.md).
Status penghapusan tersebut tidak sama dengan bukti bahwa semua artefak
historis tidak akan pernah dibutuhkan lagi.

Untuk konteks visualisasi, lihat
[catatan GSM8K](../../docs/ringkasan_visual_gsm8k_dan_cleanup_2026-08-26.md).
