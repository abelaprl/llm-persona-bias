# Penanda linguistik Gender — ETHICS

[Buka notebook utama](ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb) · [Galeri](gallery.html) · [Gambar dan CSV](figures/)

Pembaruan 2 Oktober 2026 sudah terpasang di notebook utama dan galeri. Urutan baca: cakupan, perbandingan langsung **Man − Woman**, lalu profil dan analisis pendukung. Empat panel E/I1/I2/I3 serta enam warna model mengikuti Gender GSM8K. Notebook GSM8K tidak diubah.

Empat grafik perbandingan langsung kini memakai warna penuh dan titik lebih besar. Segitiga menandai p Holm < 0,05; lingkaran menandai hasil lainnya. Ekspor gambar baru hanya **PNG 300 dpi**, tanpa SVG; CSV tetap tersedia untuk angka. SVG historis tidak diperbarui dan bukan versi visual terkini. Salinan sebelum perubahan kontras tersimpan di `_arsip/sebelum_kontras_png_20261002_193406/`.

Notebook dan `figure_index.csv` menampilkan **13 grafik**: lima grafik baru (cakupan dan empat perbandingan langsung) serta delapan grafik pendukung. Gambar historis di `figures/` tetap disimpan; gunakan indeks atau galeri aktif untuk memilih gambar terkini. Grafik padat memakai format landscape; jangan memperkecil menjadi lebar 16 cm tanpa memeriksa keterbacaan.

Seluruh **41 marker** tercakup pada profil dan perbandingan langsung. Notebook juga menyediakan penjelajah tabel untuk semua marker, skenario dan subset, serta tautan respons utuh. Sumber statistik tetap `gender_lm_v2_candidate_20260916_r3`; tidak ada perubahan parser, kamus, data, atau p-value.

Status **candidate**, bukan hasil audit manusia yang sudah tuntas. Analisis memakai penjelasan final, termasuk abstain berpenjelasan. Teks tidak tersedia tetap NA, bukan nol marker. Keragaman hanya jumlah kategori kamus berbeda, bukan seluruh kekayaan bahasa. Visual langsung memakai **Man−Woman**, sama seperti GSM8K; positif berarti lebih tinggi pada Man. Kolom sumber tetap Woman−Man, sementara kolom `display_*` mencatat transformasi tanda. Density memakai matched rank-biserial; marker biner memakai selisih prevalence dalam poin persentase dan CI 95% marginal. Median ditampilkan hanya bila keenam model mempunyai estimasi tersedia.

- [Rencana, acuan yang diperiksa, dan catatan verifikasi](../../../../docs/gender_ethics_reference_redesign_20260920.md).
- [Bundle lima visual utama terkini, PNG](../../../../result/ETHICS/reports/gender_lm_bab6_viz_20261002_r4/).
- [Bundle pendukung](../../../../result/ETHICS/reports/gender_lm_reference_presentation_20260920_r3/).
- [Salinan sebelum pemasangan visual Bab VI](./_arsip/sebelum_bab6_viz_20261002_192649/), dengan hash verifikasi file asli.
- [Arsip seluruh gambar lama](./_arsip/pembersihan_figures_20260920/figures/): 386 berkas dapat dipulihkan.
- [Arsip notebook/galeri/indeks lama](./_arsip/pembersihan_figures_20260920/presentation/).

Reproduksi lima visual utama dari akar repositori, selalu memakai direktori output baru:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/MAIN_PARSING/ETHICS/build_gender_ethics_lm_bab6_viz.py --output result/ETHICS/reports/gender_lm_bab6_viz_reproduction_01
```

Notebook dapat dijalankan seluruhnya tanpa API. Nama file notebook dipertahankan agar lokasinya tetap sama; tag v1_5 pada nama merupakan nama historis, bukan versi sumber statistik.
