# Panduan scripts

Folder ini memuat builder, audit, dan utilitas eksperimen. Nama versi lama
tidak berarti kode tidak dipakai: builder baru masih mengimpor helper lama.

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

Kelima builder masih mengimpor nama package
`src.MAIN_PARSING.LINGUISTIC_MARKERS`, sementara folder kerja library berada
di `src/MAIN_PARSING/Linguistic Marker Builder/`. Karena itu, pemanggil perlu
menyediakan alias kompatibilitas seperti loader yang digunakan pipeline
ETHICS. Cleanup ini tidak mengubah mekanisme impor builder tersebut.

Generator volcano dan perender keterbacaan mempunyai tugas berbeda:
generator menyiapkan tabel titik beserta notebook; perender membaca tabel
tersebut untuk tata letak yang lebih mudah dibaca. Keduanya dipertahankan.

## Perapian 15 September 2026

Dua fungsi visual lama yang tidak dipanggil oleh `main()` beserta lima
konstantanya dibuang dari `render_gsm8k_marker_readable_v1.py`.
Fungsi-fungsi pada alur aktif tetap sama. Tidak ada penghapusan file tambahan.

Tiga belas path di folder ini sudah berstatus penghapusan staged sebelum
cleanup. Daftar dan bukti pemeriksaan tersedia pada
[laporan cleanup scripts](../docs/cleanup_scripts_20260915_232211/README.md).
Status penghapusan tersebut tidak sama dengan bukti bahwa semua artefak
historis tidak akan pernah dibutuhkan lagi.

Untuk konteks visualisasi, lihat
[catatan GSM8K](../docs/ringkasan_visual_gsm8k_dan_cleanup_2026-08-26.md).
