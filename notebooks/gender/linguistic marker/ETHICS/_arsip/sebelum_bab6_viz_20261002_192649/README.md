# Penanda linguistik Gender — ETHICS

[Buka notebook utama](ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb) · [Galeri](gallery.html) · [Gambar dan CSV](figures/)

Presentasi baru mengikuti alur Religion/Regional dan bentuk chart Gender GSM8K: profil marker, perbandingan berpasangan, lalu variasi kategori dan konteks. Enam warna model berada dalam satu panel; Man/Woman serta E/I1/I2/I3 dibandingkan dalam gambar yang sama.

Folder `figures` hanya berisi **12 PNG, 12 SVG, dan 12 CSV** baru. Tidak ada subfolder r1/r2 atau preview `ringkas`/`model_persona` aktif. Grafik padat memakai format landscape; jangan memperkecil menjadi lebar 16 cm tanpa memeriksa keterbacaan. Font DejaVu Sans konsisten.

Seluruh **41 marker** tercakup pada profil dan perbandingan langsung. Notebook juga menyediakan penjelajah tabel untuk semua marker, skenario dan subset, serta tautan respons utuh. Sumber statistik tetap `gender_lm_v2_candidate_20260916_r3`; tidak ada perubahan parser, kamus, data, atau p-value.

Status **candidate**, bukan hasil audit manusia yang sudah tuntas. Analisis memakai penjelasan final, termasuk abstain berpenjelasan. Teks tidak tersedia tetap NA, bukan nol marker. Keragaman hanya jumlah kategori kamus berbeda, bukan seluruh kekayaan bahasa. Arah perbandingan langsung **Woman−Man**, mengikuti sumber ETHICS, berbeda dari arah chart GSM8K.

- [Rencana, acuan yang diperiksa, dan catatan verifikasi](../../../../docs/gender_ethics_reference_redesign_20260920.md).
- [Bundle reproduksi](../../../../result/ETHICS/reports/gender_lm_reference_presentation_20260920_r3/).
- [Arsip seluruh gambar lama](./_arsip/pembersihan_figures_20260920/figures/): 386 berkas dapat dipulihkan.
- [Arsip notebook/galeri/indeks lama](./_arsip/pembersihan_figures_20260920/presentation/).

Reproduksi dari akar repositori, selalu memakai run-id baru:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/MAIN_PARSING/ETHICS/build_gender_lm_reference_presentation.py --run-id reproduction_01
```

Notebook dapat dijalankan seluruhnya tanpa API. Nama file notebook dipertahankan agar lokasinya tetap sama; tag v1_5 pada nama merupakan nama historis, bukan versi sumber statistik.
