# Linguistic marker Gender ETHICS

Notebook utama:
[ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb](ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb).
Nama berkas tetap v1.5; statistik tetap **v2.0.1 candidate r3**.

## Penyajian aktif — model dan persona, 20 September 2026

Notebook utama kini memakai 12 figur. Dalam setiap panel perbandingan,
keenam model muncul bersama sebagai titik berwarna; panel memisahkan E/I1/I2/I3
dan perbandingan Woman−Man, persona−Neutral, atau implicit−explicit.

- [Galeri aktif](gender_ethics_model_persona.html).
- `figures/model_persona_01` sampai `model_persona_12`: PNG, SVG, dan CSV
  langsung di folder `figures` utama, tanpa subfolder tambahan.
- `model_persona_figure_index.csv`: indeks figur aktif.
- `model_persona_manifest.json`: sumber, hash, jumlah titik/NA, dan CI.
- Penanda numerik memakai rank-biserial yang sudah tersedia; presence memakai
  selisih prevalensi dan CI nominal. Statistik tidak dihitung ulang.
- Notebook sebelumnya dicadangkan di `_arsip/sebelum_model_persona_20260920/`.
  Grafik detail lama tetap tersedia sebagai lampiran. PNG/CSV `ringkas_*`
  yang sudah dihapus tidak dibuat ulang.

Jalankan **Run All** pada notebook utama untuk membaca ulang gambar lokal.
Paket visual aktif dibangun dengan
`src/MAIN_PARSING/ETHICS/build_gender_lm_model_persona.py`; installer menolak
menimpa tujuan yang sudah ada tanpa penanganan versi baru secara eksplisit.

## Riwayat penyajian sebelumnya

Instruksi builder di bawah menjelaskan versi sebelumnya, bukan perintah
untuk memperbarui penyajian titik model yang sekarang aktif.

- `figures/`: 88 pasang PNG/SVG utama dan 80 pasang tambahan `diagnostic_*`
  (total 168 PNG + 168 SVG), dengan gaya visual referensi GSM8K v3:
  DejaVu Sans, label model singkat, biru/magenta, dan panel forest per skenario.
- `figure_index.csv`: judul, lokasi gambar, sumber tabel, dan hash PNG.
- `presentation_source.json`: sumber gambar terpisah dari sumber statistik.
- Notebook menampilkan gambar lokal dan membaca statistik dari
  `result/ETHICS/reports/gender_lm_v2_candidate_20260916_r3/` di root repositori.
  Kolom `source_csv` pada indeks mengikuti lokasi relatif terhadap bundle tersebut.

Jalankan **Run All** untuk memeriksa sumber, menampilkan seluruh gambar utama,
ringkasan diagnostik, dan halaman marker pilihan.
Untuk membangun ulang penyajian beserta salinan gambar, jalankan dari root:

```powershell
.\.venv\Scripts\python.exe src/MAIN_PARSING/ETHICS/update_gender_lm_notebook_r3.py
```

Builder menyimpan cadangan notebook dan bukti verifikasi pada folder hasil baru.
Salinan gambar yang berbeda dari bundle presentasi terpilih ditolak; gambar
sumber r3 tidak ditimpa. CSV sumber setiap figur identik dengan r3, tetapi
gambar dirender ulang. Versi visual sebelumnya disimpan di
`_arsip/visual_r3_sebelum_gsm8k_20260918/`.

Untuk merender ulang seluruh figur ke folder baru (jangan memakai folder yang
sudah ada):

```powershell
.\.venv\Scripts\python.exe -B -u src/MAIN_PARSING/ETHICS/restyle_gender_lm_gsm8k.py --output-dir result/ETHICS/reports/gender_lm_gsm8k_style_rerun01
```

Penyelarasan visual tidak mengubah angka, arah Woman−Man, atau status candidate.
Audit manusia dan target precision/recall 99% tetap belum selesai; lihat
[validasi r3](../../../../docs/validasi_ethics_gender_lm_v2_r3_20260916.md).

## Pelengkap diagnostik 19 September 2026

Buka **bagian 9–11** pada notebook:

- Konteks rujukan gender per respons: any-gender, pengguna eksplisit,
  kutipan/hipotetis, unknown, dan flag overlap skenario. Kategori boleh overlap.
- Dukungan Woman−Man: n_pairs dan n_changed, dengan low-support terlihat.
- Sensitivitas overall versus empat subset untuk seluruh 41 marker.
- Count versus density pada sumbu terpisah untuk 30 marker density;
  sebelas marker binary tetap memakai presence.

Ubah `DIAGNOSTIC_MARKER = "stance_hedge"` pada bagian 11 dan jalankan sel itu
untuk memilih marker lain. Seluruh halaman, bukan hanya contoh yang tertanam,
dapat dibuka melalui [galeri diagnostik](diagnostic_gallery.html) atau
[indeks tambahan](diagnostic_figure_index.csv). PNG/SVG berada langsung di
`figures/`; tabel sumber tambahannya berada di `diagnostic_tables/`.
Konfigurasi sumber tercatat pada `diagnostic_source.json`.

Untuk menghasilkan ulang paket diagnostik pada path baru:

```powershell
.\.venv\Scripts\python.exe -B -u src/MAIN_PARSING/ETHICS/build_gender_lm_diagnostics.py --output-dir result/ETHICS/reports/gender_lm_diagnostics_rerun01
```

Builder notebook di atas juga memasang dan memeriksa paket diagnostik yang
dipilih pada konfigurasi. Seluruh effect, CI, p Holm dan jumlah pasangan
berasal dari r3. Reference dihitung per respons dari flag otomatis yang
sudah ada, tanpa membaca respons mentah atau mengubah detector.
`unknown`, NA, dan hasil nonsignifikan tetap ditampilkan. Tambahan ini tidak
menyelesaikan audit manusia atau menambahkan tujuh calon kategori marker.
