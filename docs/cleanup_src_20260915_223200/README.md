# Cleanup source 15 September 2026

## SPEC

Tujuan: merapikan source yang dibahas dalam inventaris Git dan menghapus
skrip sekali pakai yang pemindahan arsipnya sudah selesai. Cakupan keputusan
penghapusan adalah 139 file untracked di `src/` pada snapshot awal cleanup.
Tidak ada eksperimen atau pemrosesan dataset; cakupan dataset tidak berlaku.

## DESIGN PLAN

Pemeriksaan membaca isi kandidat, mencari referensi nama pada kode,
dokumentasi, dan sumber sel notebook, kemudian membandingkan hash tujuan
arsip dengan catatan pemindahan. Skrip reproduksi dipertahankan, termasuk
entry point tanpa referensi masuk. Struktur impor tetap pada lokasi yang ada.

## IMPLEMENTATION

Empat file yang dipilih untuk penghapusan, semuanya di
`src/MAIN_PARSING/ETHICS/`:

| File | Bukti tugas pemindahan telah selesai |
|---|---|
| `archive_regional_ethics_notebooks.ps1` | 16/16 hash tujuan cocok dengan `notebooks/archive/20260910_regional_ethics/archive_index.csv` |
| `archive_superseded_root_notebooks.ps1` | 7/7 hash tujuan cocok dengan `notebooks/archive/20260910_superseded_notebooks/archive_index.csv` |
| `archive_unused_notebook_assets.py` | 107/107 hash tujuan cocok dengan `docs/ethics_archive_plan_20260910.json` |
| `move_unused_notebook_assets.ps1` | Memakai catatan 107 file yang sama; memerlukan path sumber lama dan tujuan yang belum ada |

Keempat skrip mematok arsip tanggal 10 September, dan tidak ditemukan referensi
nama skripnya dalam 661 file yang terbaca pada cakupan scan. Ketiga catatan
arsip beserta seluruh isi arsip tetap di lokasi semula.

Dua path sumber pada catatan 107 file kini berisi gambar dengan hash berbeda:
`notebooks/gender/ethics accuracy analysis/figures/01_ethics_absolute_accuracy_five_categories.png`
dan pasangannya `.svg`. Keduanya dipertahankan; salinan arsip tetap cocok
dengan hash historis. Total 128/130 path sumber lama sudah tidak ada.

Perapian navigasi tersedia pada
[SOURCE_GUIDE.md](../../src/MAIN_PARSING/ETHICS/SOURCE_GUIDE.md), yang ditautkan
dari README parsing.

## CHECKLIST

- [x] Keputusan untuk 139 file dicatat di [inventory.csv](inventory.csv).
- [x] Keempat kandidat dibaca penuh dan seluruh 130 hash tujuan arsip cocok.
- [x] Salinan empat source disimpan di [removed_sources.zip](removed_sources.zip)
  sebelum penghapusan; isi ZIP diverifikasi terhadap SHA256 snapshot.
- [x] Skrip analisis, audit, pengujian, dan reproduksi dipertahankan.
- [x] Verifikasi setelah penghapusan lulus: lihat [verification.json](verification.json).

## TESTING

Verifikasi cleanup memeriksa bahwa empat path sudah tidak ada, seluruh file
source lain pada snapshot tetap identik kecuali README yang sengaja diberi
tautan, hash staging tidak berubah, ZIP valid, dan tautan panduan tersedia.
Hasil: empat path terhapus, 481 file source lain identik byte demi byte,
ZIP pemulihan valid, staging identik, dan seluruh 14 tautan panduan tersedia.
`git diff --check -- src/MAIN_PARSING/ETHICS/README.md` juga lulus.
Tidak dijalankan API atau regenerasi angka penelitian untuk penghapusan
skrip migrasi yang tidak dirujuk ini; suite analisis tidak dijalankan ulang.

## INTEGRATION

Gunakan panduan entry point untuk memilih pipeline. Inventaris menyertakan
alasan mempertahankan tiap source; ketiadaan referensi tidak dipakai sebagai
satu-satunya alasan penghapusan. Tidak dilakukan stage, commit, atau push.

ZIP adalah salinan pemulihan source untracked, bukan komponen runtime.
Untuk memulihkan skrip tertentu, baca path di dalam ZIP dan ekstrak hanya
file yang diperlukan ke path asal yang belum ada. Jangan menimpa versi baru.

## Batas pemeriksaan

Pencarian referensi adalah pencarian statis nama file/modul, bukan bukti
berapa kali program pernah dijalankan. Scan mencakup ekstensi `.py`, `.ps1`,
`.sh`, `.md`, `.ipynb`, `.json`, dan `.cjs` di `src/`, `scripts/`, `docs/`,
dan `notebooks/`; isi output notebook tidak dipakai sebagai dependensi.
Inventaris Git lama dikecualikan. File lebih dari 20 MB dilewati; rincian
file yang dilewati dan pesan akses tersedia di [audit.json](audit.json).
Folder `src/MAIN_PARSING/ETHICS/tests/pooled_test_93c4zanf/` tidak dapat dibaca
karena izin akses, sehingga isinya belum diperiksa. Pemanggil di luar
repositori dan path yang dirangkai secara dinamis tidak dapat dipastikan
dengan scan ini. Source di luar daftar untracked tidak dipilih untuk dihapus.
