# Rencana revisi gambar ETHICS regional untuk Bab VI — 14 cm

Tanggal: 9 Oktober 2026. Status: rencana dan handoff; rendering belum dikerjakan.
Permintaan pengguna: buat rencana penyetaraan tampilan ETHICS regional dengan revisi gender/religion dan simpan di D:\TugasAkhir.
Dokumen ini bukan perintah untuk menjalankan eksperimen baru atau mengubah naskah tesis sekarang.

## 1. SPEC

### Tujuan dan kondisi yang sudah diperiksa

Siapkan gambar ETHICS regional yang terbaca pada lebar cetak 14 cm, mengikuti prinsip pemecahan panel paket religion. Tinggi gambar boleh berbeda. Menyamakan tampilan tidak berarti menyamakan definisi metrik yang berbeda.

- Sumber aktif: `notebooks/regional/ethics accuracy analysis/`.
- `LATEST_RUN.json` menunjuk `result/analyses/accuracy/runs/regional_ethics_presentation_20261006_083755_469196/`.
- Pada pemeriksaan 9 Oktober, naskah `D:\ProposalTA-18222008\Bab VI - ETHICS Regional Akurasi.tex` masih merujuk 10 PNG di `image/BabVI/ethics/regional_kultural/notebook_20261002/`. Sembilan berbeda hash dari sumber 6 Oktober; gambar token 14 identik. Perbedaan hash membuktikan berkas berbeda, bukan perubahan statistik.
- Gambar sumber 06 masih 17 kolom, gambar efek 11 masih 16 kolom, dan gambar alasan 05 masih enam model sekaligus. Belum mengikuti pemecahan panel religion.
- Pembahasan regional sudah membatasi akurasi pada Commonsense/Justice dan menafsirkan Deontology/Virtue sebagai distribusi keputusan. Pertahankan pembatasan ini.

### Acuan dan input

1. Baca `AGENTS.md`, `CLAUDE.md`, `docs/instruction.md`, serta petunjuk lokal tambahan jika ada saat implementasi.
2. Acuan tampilan: `result/analyses/accuracy/runs/religion_ethics_bab6_14cm_20261008_230544_472975/`, khususnya `MULAI_DI_SINI.md`, `CAPTIONS.md`, `bab6/`, dan `proof/`.
3. Riwayat regional: `docs/PENYELARASAN_AKURASI_REGIONAL_GENDER_RELIGION_2026-10-06.md` dan README notebook aktif.
4. Tabel regional di `notebooks/regional/ethics accuracy analysis/tables/`:
   - `ethics_regional_parity_condition_summary.csv`: total, nonmissing, committed, abstain, missing, n_correct, commit_rate, accuracy_committed, distribusi keputusan.
   - `ethics_regional_parity_paired_effects_vs_neutral.csv`: pasangan, delta, p mentah, p Holm, family_size, Cohen's h.
   - `ethics_regional_parity_abstain_reasons.csv`: reason_count, abstain, nonmissing, pct_of_abstain, pct_of_nonmissing.
   - Tabel kategori/status lain yang dirujuk notebook; cocokkan ID dan penyebut sebelum menggabungkan.
5. Notebook utama `ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb` dan pendamping `ETHICS_Regional_Accuracy_Heatmap_06_13.ipynb` sebagai rujukan implementasi yang ada.

Cakupan: seluruh enam model, empat kawasan, E/I1/I2/I3 dan N yang tersedia pada tabel beku; tanpa sampling data dan tanpa API. Ini pekerjaan presentasi, bukan audit ulang parser/pairing atau pengujian ulang.

## 2. DESIGN PLAN

### Pembagian identitas dan format

- Kelompok A: Amerika Utara dan Asia Timur (`north_american`, `east_asian`).
- Kelompok B: Asia Selatan dan Afrika Sub-Sahara; verifikasi ID persis dari tabel, jangan membangun ID dari label tampilan.
- Satu kelompok deskriptif: 8 kolom persona (2 kawasan x E/I1/I2/I3) + N = 9 kolom.
- Satu kelompok efek persona terhadap N: 8 kolom, tanpa kolom N bernilai nol buatan.
- N yang diulang pada panel adalah referensi visual, bukan observasi baru. Deduplicasi N saat memeriksa total.
- Pertahankan urutan model, warna identitas, nama model, dan makna warna status dari sumber. Nama kawasan boleh menggunakan AU/AT/AS/ASS jika dijelaskan pada gambar/caption.
- Ekspor PDF vektor dan PNG minimal 300 dpi. Lebar akhir PDF 14 cm; periksa ukuran aktual setelah ekspor karena tight bounding box bisa mengubah ukuran.
- Sasaran ukuran teks efektif pada lebar 14 cm: angka dan label utama minimal 8 pt. Jika tidak muat, pecah lagi; jangan mengecilkan teks agar muat. Semua pasangan gambar memakai skala warna sama untuk metrik yang sama.

### Paket prioritas — 10 gambar, mengikuti religion

| Isi | Jumlah awal | Desain dan makna |
|---|---:|---|
| Akurasi committed (sumber 06) | 2 | Satu gambar per kelompok kawasan; panel Commonsense/Justice; 9 kolom; angka akurasi dan coverage c bersama. c = 100 x committed/total, bukan committed/nonmissing. |
| Efek akurasi persona terhadap N (sumber 11) | 2 | Satu per kelompok; Commonsense/Justice; 8 kolom; risk_difference_pp pada pasangan sama-sama committed; bintang dari keputusan Holm sumber. |
| Empat status respons (sumber 00) | 2 | Satu per kelompok; penyebut seluruh respons; wrong, acceptable, abstain, missing. Jika menampilkan I gabungan, jumlahkan hitungan I1-I3 dahulu lalu bagi penyebut gabungannya. |
| Komposisi alasan abstain (sumber 04/tabel alasan) | 4 | Dua kelompok kawasan x dua kelompok model (masing-masing 3 model, mengikuti urutan sumber); 9 kolom; skala tetap 0–100%; tampilkan n abstain per model-skenario. |

Jumlah adalah rancangan awal, bukan batas yang mengalahkan keterbacaan. Simpan data setiap sel dan metadata penyebut ke CSV agar dapat ditelusuri.

### Bedakan dua ukuran alasan abstain

- Gambar utama yang mengikuti religion memakai `pct_of_abstain = 100 x reason_count / abstain` dan n = abstain. Kategori alasan berasal dari sumber; periksa apakah kategori saling eksklusif sebelum menyatakan persentasenya berjumlah 100%.
- Gambar 05 yang kini dirujuk tesis memakai `pct_of_nonmissing = 100 x reason_count / nonmissing`. Ini ukuran berbeda dan tetap dipertahankan sebagai pelengkap dengan judul, penyebut, serta n yang sesuai.
- Jangan mengganti nilai gambar 05 dengan persentase dari abstain sambil mempertahankan nama/caption lama. Buat aset baru dan catat perubahan ukuran pada mapping integrasi.
- Penyebut nol ditampilkan NA dengan n=0; tidak diganti 0%. Persentase tinggi pada n kecil harus terlihat dari anotasi n.

### Cakupan gambar lain yang sudah aktif di tesis

Inventaris seluruh 10 referensi aktif sebelum bekerja. Paket 10 gambar di atas belum menggantikan seluruh isi regional. Jangan menghilangkan hasil berikut hanya agar jumlah gambar sama dengan religion:

| Gambar aktif | Perlakuan dalam revisi |
|---|---|
| 07 keberhasilan efektif | Tetap tersedia; pecah per dua kawasan bila padat. Penyebut tetap nonmissing sebagaimana sumber. |
| 10 efek coverage | Tetap tersedia; pecah per dua kawasan, dan per pasangan subset bila empat subset terlalu tinggi. |
| 12 efek keberhasilan efektif | Tetap tersedia; pecah per dua kawasan; pasangan/penyebut mengikuti outcome sumber. |
| 08 stance Deontology/Virtue | Tetap tersedia; pecah per dua kawasan bila perlu; jangan diberi label akurasi. |
| 13 efek stance Deontology/Virtue | Tetap tersedia; pecah per dua kawasan; pertahankan statistik sumber. |
| 14 token | Pertahankan aset identik, tanpa perhitungan ulang. Catat status keterbacaannya pada proof. |
| 05 prevalensi alasan/nonmissing | Pertahankan sebagai ukuran berbeda dari komposisi alasan/abstain; buat versi terbaca bila diperlukan. |

Distribusi wrong/acceptable pada committed boleh disediakan terpisah di `pelengkap_committed/`, sebagaimana religion; jangan campur penyebutnya dengan empat status dari seluruh respons. Semua penempatan Bab VI/lampiran adalah usulan: tidak memindahkan atau menghapus gambar dari naskah tanpa instruksi implementasi tersendiri.

### Invarian angka dan inferensi

- Jangan mengubah respons, parser, scoring, dataset, persona, jumlah sampel, pemilihan pasangan, p-value, atau keputusan signifikansi.
- Akurasi hanya Commonsense/Justice, selalu disertai coverage. Deontology/Virtue tetap distribusi keputusan, bukan akurasi moral.
- Nilai efek persona–N di heatmap 10–13 adalah risk difference dalam poin persentase. Jangan menamainya Cohen's h; h tetap kolom terpisah pada tabel jika diperlukan.
- Koreksi Holm tetap 16 kontras per model/subset/outcome, meskipun dibagi menjadi dua gambar berisi 8 kolom. Tidak melakukan koreksi ulang per gambar dan tidak menambahkan prasyarat Q Cochran.
- Pertahankan arah persona minus N dan aturan eligibilitas pasangan per outcome. Jangan mengurangkan dua akurasi marginal untuk menggantikan efek dari pasangan sama-sama committed.
- N pasangan dan discordant tersedia pada tabel pendukung; jangan disamakan dengan n abstain atau jumlah total respons.
- Pooling I dan/atau subset hanya deskriptif berbasis hitungan yang sesuai. Jangan melebur empat subset menjadi satu akurasi ETHICS.

## 3. IMPLEMENTATION — langkah untuk pelaksana berikutnya

Belum dijalankan pada saat dokumen ini dibuat.

1. Baca ulang status terkini dan inventaris input; simpan hash sumber tabel, gambar, notebook dan manifest sebelum rendering. Jangan berasumsi repo tetap sama dengan 9 Oktober.
2. Gunakan kembali renderer religion bila cocok setelah memeriksa pemetaan kolom. Jika perlu kode baru, tempatkan di `src/benchmarks/ethics/` dengan nama yang jelas untuk rendering regional 14 cm, tanpa perubahan ekstraktor.
3. Buat run baru `result/analyses/accuracy/runs/regional_ethics_bab6_14cm_<timestamp>/`. Jangan menimpa run lama atau mengganti `figures/` aktif pada tahap ini.
4. Render paket prioritas, lalu seluruh pelengkap yang diperlukan agar referensi lama tetap terwakili. Tampilkan progres per gambar dan catat kegagalan/NA.
5. Hasil run memuat `bab6/`, `pelengkap/`, `pelengkap_committed/` bila digunakan, `tables/`, `proof/`, `MULAI_DI_SINI.md`, `CAPTIONS.md`, `manifest.json`, `checks.json`, dan `mapping_integrasi.csv`.
6. Manifest mencatat sumber, hash, metrik, subset, kelompok persona/model, unit, penyebut, ukuran fisik, dan perintah reproduksi. Sertakan versi dependensi yang benar-benar digunakan.
7. Mapping integrasi memuat label/path .tex lama, path gambar baru (boleh satu-ke-banyak), perubahan caption/penyebut, usulan lokasi, serta status validasi. Gambar yang tidak dirender ulang harus diberi status eksplisit, bukan hilang dari inventaris.

## 4. CHECKLIST

- [x] Sumber aktif, acuan religion, dan referensi tesis telah diperiksa untuk menyusun rencana.
- [x] Pembagian panel dan perbedaan penyebut alasan abstain ditentukan.
- [x] Invarian statistik dan keluaran handoff ditentukan.
- [ ] Renderer dan paket gambar baru selesai.
- [ ] Seluruh nilai/penyebut/p-value yang ditampilkan cocok dengan tabel sumber.
- [ ] Hash sumber sebelum–sesudah identik.
- [ ] Proof A4 seluruh gambar dan caption diperiksa secara digital.
- [ ] Cetak fisik diperiksa (jika belum, tulis belum; jangan mengklaim sudah).
- [ ] Integrasi .tex dan kompilasi TA.pdf — pekerjaan lanjutan, bukan bagian penulisan rencana ini.

## 5. TESTING — kriteria penerimaan implementasi

1. Cocokkan angka plot secara terprogram ke tabel beku berdasarkan ID model/persona/subset/outcome; toleransi hanya pembulatan tampilan. Periksa keunikan baris dan kelengkapan seluruh skenario, termasuk N.
2. Verifikasi rumus akurasi, coverage, kedua jenis persentase alasan, serta distribusi status dari hitungan sumber. Pastikan tidak mencampur angka 0–1 dengan 0–100 atau persen dengan poin persentase.
3. Cocokkan risk_difference_pp, Holm, arah efek, family_size=16, dan jumlah hasil signifikan sebelum–sesudah pemisahan. Ini pemeriksaan preservasi, bukan menghitung ulang uji.
4. Pastikan penjumlahan status memakai penyebut tepat; pooling I menggunakan hitungan. Hitung total unik sebelum menggandakan N untuk tampilan.
5. Decode semua PNG dan buka PDF. Periksa tidak ada anotasi, label panjang, legenda, colorbar, atau tanda bintang yang terpotong/bertumpuk.
6. Buat proof A4 dengan lebar gambar aktual 14 cm dan margin/area teks mengikuti tesis. Sertakan caption utuh; jangan menggunakan caption terpotong atau elipsis sebagai bukti gambar muat di halaman. Pisahkan halaman jika tinggi gambar dan caption melebihi area cetak.
7. Buka seluruh proof pada ukuran halaman, terutama abstain. Nilai keterbacaan tanpa mengandalkan zoom. Catat halaman/gambar yang diperiksa dan keterbatasan inspeksi digital.
8. Hash semua sumber/hasil lama harus identik. Catat perintah reproduksi yang berhasil dan hasil checks.json. Jangan menandai selesai hanya karena file berhasil dibuat.

## 6. INTEGRATION — keluaran untuk repo tesis

Serahkan path `MULAI_DI_SINI.md`, caption siap pakai, indeks PNG/PDF, proof A4, dan mapping semua referensi lama ke aset baru. Ringkasan harus membedakan: gambar selesai, angka tervalidasi terhadap tabel beku, proof digital selesai, cetak fisik belum/sudah, serta .tex belum/sudah diubah.

Pada tahap implementasi tesis berikutnya, salin aset tervalidasi ke folder baru `image/BabVI/ethics/regional_kultural/bab6_14cm_<tanggal>/`, sesuaikan caption dan pembahasan yang penyebutnya berubah, lalu kompilasi dan periksa halaman TA.pdf. Jangan menganggap memperbarui folder gambar sumber otomatis memperbarui naskah/PDF.

Risiko utama: teks masih kecil walau PNG beresolusi tinggi; persentase abstain dan nonmissing tertukar; N terhitung dua kali; keluarga Holm terpecah mengikuti layout. Strategi pemeriksaan di atas menargetkan keempat risiko tersebut. Tampilan yang seragam tidak memvalidasi ulang parser, desain persona, ataupun pairing historis.
