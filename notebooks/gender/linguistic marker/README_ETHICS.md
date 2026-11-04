# Linguistic marker Gender ETHICS

Lokasi aktif, diperbarui 2 Oktober 2026:
[notebook ETHICS](<ETHICS/ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb>)
berada di folder `ETHICS/`. Notebook dan [galeri aktif](ETHICS/gallery.html)
menampilkan 26 PNG dari 16 kelompok grafik, selebar 14 cm dengan font minimal
8 pt. Gambar padat mempunyai halaman lanjutan `_p02`, `_p03`, dan seterusnya;
notebook menampilkan semuanya. Versi sebelumnya tersimpan di `_arsip`.
Isi notebook memakai v2.0.1 candidate r3; nama berkas v1.5 dipertahankan.
Visual utama memakai arah **laki-laki − perempuan**, empat panel skenario, dan warna model
yang konsisten dengan GSM8K v3. Statistik sumber ETHICS r3 tetap utuh;
pembalikan tanda hanya dicatat pada kolom tampilan. Profil, dukungan pasangan,
keragaman kategori, dan konteks rujukan tetap tersedia sebagai pendukung.

Grafik terhadap netral kini ditambahkan sebelum perbandingan langsung, mengikuti
grafik GSM8K Bab VI utama: laki-laki/perempuan × E/I1/I2/I3, arah persona−netral,
warna kelompok marker, sumbu x rank-biserial dan sumbu y −log10(p Holm).
Nama gambar dimulai dengan `N01_`, `N02_`, dan `N03_`; keluaran gambar hanya PNG.
[Panduan folder aktif](ETHICS/README.md) menjelaskan sumber dan cara menjalankan ulang.
Pada pembaruan konsistensi 2 Oktober 2026, notebook GSM8K juga memakai penamaan
seragam dan warna titik penuh. Gambar terpilih kedua benchmark telah dipasang
ke LaTeX Bab VI dan Lampiran D. Lihat `docs/gender_lm_consistent_20261002.md`
untuk paket sumber, cadangan, dan hasil pemeriksaan.
Catatan berikut merupakan riwayat penyajian versi sebelumnya.

## Riwayat penyajian 9–13 September 2026

Tambahan penyajian 13 September 2026 sudah dipasang pada notebook utama:
panduan membaca effect size/CI, seluruh perbandingan dengan hasil tidak
signifikan, grafik langsung jumlah kata per subset, heterogenitas arah, dan
contoh respons berpasangan. Notebook kini 54 sel; seluruh 44 sel lama beserta
outputnya tetap utuh. [Buka penjelajah tabel](../../../result/benchmarks/ethics/reports/presentation_completion_20260913_r5/gender/explorer.html)
atau [catatan validasi](../../../docs/ethics_lm_presentation_completion_20260913.md).
Status tetap candidate. Jumlah sel di catatan 9–10 September di bawah adalah
status historis sebelum tambahan ini.

Notebook utama: `ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb`.
Notebook GSM8K di folder ini tetap terpisah dan tidak diubah oleh pipeline ETHICS.

## Urutan perbandingan

Presentasi 10 September 2026 menempatkan **Laki-laki dan Perempuan** sebagai
hasil utama, lalu **Terhadap Netral** dan **Implisit dan Eksplisit** sebagai
pendukung. Urutan ini berlaku pada bagian per subset dan ETHICS Gabungan.
E, I1, I2, dan I3 tetap terpisah. Perbedaan signifikansi masing-masing persona
terhadap N tidak digunakan untuk menyimpulkan perbedaan Man–Woman.

Ringkasan baru mencakup semua model, jumlah perbandingan/signifikansi, arah efek,
dukungan pasangan, serta contoh hasil per model dan skenario. Grafik arah
menampilkan semua hasil, termasuk yang tidak signifikan. Kesamaan arah
antarmodel bersifat deskriptif, bukan uji gabungan antarmodel.

Tabel dan grafik presentasi baru:
`result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_man_woman_20260910_r2/`.
Salinan grafik tersedia di `figures/gender_ethics_man_woman_20260910_r2/`.
Folder `figures/` sekarang hanya memuat versi aktif tersebut. Versi sebelumnya
dipindahkan ke `_arsip/`, bukan dihapus permanen. Empat grafik pendukung
gabungan (terhadap netral dan implisit–eksplisit) ikut tersedia di folder aktif.
Seluruh angka uji berasal dari dua bundle di bawah, tanpa penghitungan ulang.
Detail revisi: `docs/ethics_gender_man_woman_20260910.md`.
Verifikasi final: 23 sel kode tanpa error, 38 ID sel lama tetap ada, dan
4.320 baris tabel presentasi cocok dengan hasil sumber. Notebook kini memiliki
44 sel. Sepuluh grafik arah baru dan sepuluh salinan grafik utama tersedia
sebagai 40 file PNG/SVG di folder presentasi tersebut. Setelah perapian folder,
salinan lokal memuat tambahan 8 file grafik pendukung (total 48 file).

Untuk mereproduksi presentasi, salin `presentation_source.ipynb` dari bundle
presentasi ke path notebook **baru**, kemudian jalankan:

```powershell
.\.venv\Scripts\python.exe -u src/MAIN_PARSING/ETHICS/present_gender_ethics_man_woman.py --output-dir result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_man_woman_rerun01 --notebook "notebooks/gender/linguistic marker/ETHICS_Gender_LM_presentation_rerun01.ipynb"
```

Perintah mengeksekusi seluruh notebook yang telah disusun ulang, menyimpan
backup, dan membuat folder grafik baru. Output yang sudah ada ditolak.
`presentation_source.ipynb` menyimpan sumber 38 sel sebelum penataan;
`notebook_before.ipynb` menyimpan target tepat sebelum run terbaru. Keduanya
tidak selalu sama. `--source-notebook` dapat membaca sumber 38 sel tersebut
secara terpisah dari notebook target.

## Bundle statistik

Bundle run yang ditampilkan notebook:
`result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_workspace_v1_5_20260909_r2/`.
Validasi otomatis lulus; 11/11 sel kode telah dieksekusi tanpa error.
Bundle menyediakan 27 grafik PNG/SVG. Catatan verifikasi terperinci tersedia
di `docs/ethics_gender_lm_workspace_20260909.md`.

## Sumber dan cakupan

Pipeline menggunakan marker per respons yang sudah dibekukan di
`result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0/`.
Scoring jawaban bersumber dari
`result/benchmarks/ethics/reports/Gender_Persona/_combined/parser_refinement_20260820/all_models_results_full.csv`.
Hash sumber dan implementasi dicatat dalam bundle. Pipeline ini tidak melakukan
rerun API, scoring jawaban, atau ekstraksi marker dari awal.

Analisis mencakup 21 ukuran umum (3 ukuran teks dan 18 marker), 11 marker
gender, komposisi pembuka, serta pemeriksaan deskriptif kamus 95 istilah gender.
Marker tambahan pada registry dibaca sebagai deskriptif, bukan diam-diam
ditambahkan ke keluarga pengujian utama. Definisi dan sumber setiap marker
tersedia pada `workspace_full_marker_registry.csv` dan inventaris terintegrasi.

Satu respons adalah satu observasi; pasangan berasal dari soal yang sama.
Empat subset dianalisis terpisah. Seluruh teks penjelasan yang tersedia,
termasuk *abstain*, masuk analisis utama. *Missing* dan *technical-only* tetap
NA, bukan nol. Jumlah mentah, density per 100 kata, dan presence dilaporkan.
Tidak ada angka akurasi gabungan ETHICS dalam notebook ini.

## Membangun ulang

Dari root `D:\TugasAkhir`, gunakan Python lingkungan proyek. Pilih **path baru**
untuk bundle maupun notebook setiap kali membangun ulang, misalnya:

```powershell
.\.venv\Scripts\python.exe -u src/MAIN_PARSING/ETHICS/run_gender_ethics_lm_workspace.py --bundle-dir result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_workspace_v1_5_rerun01 --notebook "notebooks/gender/linguistic marker/ETHICS_Gender_LM_rerun01.ipynb" --execute
```

Perintah membangun bundle statistik dengan 2.000 bootstrap resamples,
memvalidasi tabel dan gambar, menambahkan diagnostik, lalu mengeksekusi dan
menyimpan notebook. Path yang sudah ada ditolak. `--reuse-bundle` hanya untuk
membaca kembali bundle yang sudah selesai dan membuat notebook pada path baru.
Notebook memeriksa hash artefak sebelum menampilkan tabel dan grafik.

Uji otomatis:

```powershell
.\.venv\Scripts\python.exe -u src/MAIN_PARSING/ETHICS/run_gender_ethics_lm_workspace.py --bundle-dir result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_workspace_v1_5_20260909_r2 --test-only
```

Loader kompatibilitas hanya berlaku dalam proses Python: nama impor lama
diarahkan ke folder `src/MAIN_PARSING/Linguistic Marker Builder` yang sudah ada.
Tidak ada pemindahan atau pemulihan file sumber lama.
Runner menyimpan cache vektor numerik dan pasangan kunci untuk satu kelompok
data pada satu waktu. Pengujian membandingkan hasilnya dengan pemasangan asli,
termasuk urutan baris, nilai hilang, persona tidak tersedia, dan sensitivitas.
Rumus uji, seed, dan koreksi statistik tidak diganti.

## Batas interpretasi dan audit

Semua hasil tetap **candidate**. Template precision/recall untuk dua penilai
disediakan oleh builder; pipeline tidak mengisi label manusia. Gate hasil final
tetap tertutup sampai audit, agreement, dan adjudikasi selesai. Kelulusan
validator otomatis bukan bukti validitas semantik marker atau scoring jawaban.

Ada perbedaan implementasi statistik dengan Bab IV yang belum diselesaikan:
kode menggunakan sign test sebagai uji utama ketika proporsi selisih nol
melebihi 50%, absolute Bowley skewness melebihi 0,5, atau Wilcoxon tidak tersedia.
Bab IV menempatkan sign test sebagai pelengkap Wilcoxon. Notebook menyatakan
perbedaan ini; tidak mengubah metode lama tanpa persetujuan metodologis.
Pada tabel numerik, `ci_low`/`ci_high` merupakan bootstrap CI median selisih
berpasangan, **bukan** CI rank-biserial. Pada tabel presence, CI berlaku untuk
selisih prevalensi.

Pada grafik core, titik pekat menunjukkan `p_holm < 0.05`; titik pudar
menunjukkan hasil yang tidak melewati ambang itu. Berlian hitam adalah median
deskriptif effect size lintas titik/model, bukan estimasi efek gabungan atau
hasil uji lintas model. Effect size marker langka perlu dibaca bersama
`n_nonzero`, bukan hanya posisi titik.

Diagnostik status memakai Q Cochran pada sembilan persona berpasangan,
Holm atas 72 uji, serta rentang prevalensi dalam percentage points. Perbedaan
status tidak langsung membuktikan parser bias: perilaku abstain dan kegagalan
infrastruktur harus dibedakan. Analisis committed-only adalah sensitivitas,
bukan pengganti seluruh teks.

Perbandingan implisit juga membawa isi warmup dan tambahan riwayat dialog.
Istilah gender yang tidak muncul literal dalam soal belum tentu merujuk
pengguna. Cacat prompt deontology/virtue tidak diperbaiki oleh analisis marker.
Jangan menjadikan grafik candidate sebagai bukti final Bab VI sebelum
prasyarat ilmiah tersebut dituntaskan.

## ETHICS Gabungan

Bagian tambahan memakai bobot yang sama untuk setiap respons, bukan untuk
setiap subset. Model dan skenario tetap terpisah. `subset_weights.csv` dan
`pair_composition.csv` menunjukkan kontribusi subset dan jumlah pasangan;
TTR-300 memiliki dukungan pasangan tersendiri. Tidak ada akurasi gabungan.

Tabel dan grafik tambahan berada di
`result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_20260909_r4/`.
Pada penambahan gabungan tanggal 9 September, tujuh sel kode baru dieksekusi
tanpa error dan enam grafik ditambahkan; 23 sel semula beserta outputnya tetap
identik saat itu. Revisi presentasi tanggal 10 September menyusun ulang urutan
bagian dan menambahkan ringkasan tanpa menghapus analisis tersebut.
Sumber sel tambahan:
`src/MAIN_PARSING/ETHICS/templates/ethics_gender_pooled_cells.json`.
Backup 23 sel semula berada pada `notebook_before.ipynb` dalam folder output.

Untuk menghitung ulang, salin backup itu terlebih dahulu ke path **baru** di
`notebooks/`, lalu pilih direktori output baru. Jangan mengedit backup dalam
`result/`. Jalankan dua perintah berikut dengan path baru tersebut:

```powershell
.\.venv\Scripts\python.exe -u src/MAIN_PARSING/ETHICS/build_gender_ethics_lm_pooled.py --output-dir result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_rerun01 --notebook "notebooks/gender/linguistic marker/ETHICS_Gender_LM_rerun01.ipynb"
.\.venv\Scripts\python.exe -u src/MAIN_PARSING/ETHICS/build_gender_ethics_lm_pooled.py --output-dir result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_rerun01 --notebook "notebooks/gender/linguistic marker/ETHICS_Gender_LM_rerun01.ipynb" --append --execute-added
```

Builder menyimpan checkpoint setelah setiap filter pasangan selesai.
`--checkpoint-from` dapat membaca checkpoint lengkap dari run sebelumnya
ketika hash sumber dan kode identik; direktori output tetap harus baru.
Checkpoint belum merupakan hasil yang lolos validator akhir.

Tes tambahan:

```powershell
.\.venv\Scripts\python.exe -m unittest src.MAIN_PARSING.ETHICS.tests.test_gender_ethics_lm_pooled -v
```
