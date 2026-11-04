# Notebook linguistic marker yang dipakai

| Kegunaan | Notebook |
| --- | --- |
| ETHICS Gender: gabungan dan sensitivitas per subset | [ETHICS v2.0.1 r3](<linguistic marker/ETHICS/ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb>) |
| GSM8K Gender: Man–Woman dan pembanding skenario | [GSM8K v3](<linguistic marker/GSM8K/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb>) |
| Pendukung GSM8K: kata gender, pronomina, dan word cloud | [Kata dan pronomina](linguistic%20marker/pendukung/linguistic_marker_gender_words_pronouns.ipynb) |

GSM8K aktif sejak 4 Oktober 2026 berada di `linguistic marker/GSM8K/`,
dengan [20 PNG hasil revisi](<linguistic marker/GSM8K/figures/>) di `figures/`.
[Panduan GSM8K](<linguistic marker/GSM8K/README.md>) memuat sumber tabel dan arsip
notebook sebelumnya. Angka serta gambar LaTeX tidak berubah karena penataan ini.

Grafik ETHICS aktif berada di
`linguistic marker/ETHICS/figures/` (88 pasang PNG/SVG utama dengan gaya GSM8K v3,
ditambah gambar `diagnostic_*` untuk konteks, dukungan pasangan, dan sensitivitas;
tabel sumber tetap bundle r3, sumber gambar dicatat pada `presentation_source.json`).
Notebook berada langsung di `linguistic marker/ETHICS/`; nama berkas v1.5
dipertahankan, tetapi isi memakai v2.0.1 candidate r3.
[Galeri diagnostik Gender ETHICS](<linguistic marker/ETHICS/diagnostic_gallery.html>)
menyediakan halaman per marker; notebook bagian 9–11 memuat navigasinya.
Folder `linguistic marker/figures/gender_ethics_man_woman_20260910_r2/`
tetap menyimpan grafik historis.
Status ilmiah notebook utama tetap *candidate*; penataan folder tidak
menyelesaikan audit marker atau mengubah hasil statistik.

## File lama yang masih diperlukan

`linguistic marker/template/linguistic_marker_gender_ethics_direct_thesis_v1_2.ipynb`
dipakai sebagai `SOURCE_NOTEBOOK` oleh
`src/MAIN_PARSING/ETHICS/create_gender_ethics_direct_thesis_notebook_v1_3.py`.
File ini adalah template historis, **bukan notebook utama untuk membaca hasil**.
Rujukan pembuat notebook sudah disesuaikan dengan lokasi tersebut.

Notebook pendukung kata/pronomina tetap dipertahankan karena mempunyai tampilan
khusus yang tidak seluruhnya ditampilkan notebook utama. Kodenya tidak diubah;
untuk menjalankan ulang gunakan working directory `D:\TugasAkhir`, sesuai
cara kode lamanya mencari `result/`.

## Arsip

Pada 10 September 2026, 12 notebook lama dipindahkan ke
`linguistic marker/_arsip/notebooks_20260910/`:

- `asal_gender/`: 8 notebook yang sebelumnya langsung berada di `notebooks/gender`.
- `asal_img/`: 4 notebook yang sebelumnya berada di `notebooks/gender/img`.

Nama file dan isi notebook tidak diubah. Untuk memulihkan, pindahkan file
kembali ke folder asal sesuai dua kategori tersebut, setelah memastikan
tidak ada file bernama sama di tujuan. Dokumen historis masih dapat menyebut
path sebelum pengarsipan; data dan bundle hasil di `result/` tetap di tempatnya.

Pemeriksaan SHA-256 mencakup seluruh 335 file yang sudah ada dalam folder
`notebooks/gender` sebelum penataan: 12 notebook diarsipkan, 1 notebook
pendukung dipindahkan, dan semua file lainnya tetap identik. Notebook utama,
notebook akurasi dan grafik di `img` tidak diubah. Setelah pemeriksaan awal,
template v1.2 juga dipindahkan ke `linguistic marker/template/`; SHA-256-nya
tetap identik. Tidak ada notebook linguistic marker yang tersisa di luar
folder `linguistic marker` dalam `notebooks/gender`.

`ETHICS_Gender_Persona_Publication_Analysis_v4_Gabungan_Deskriptif.ipynb`
adalah notebook **akurasi**, bukan linguistic marker. File tersebut tidak
termasuk pengarsipan ini.
