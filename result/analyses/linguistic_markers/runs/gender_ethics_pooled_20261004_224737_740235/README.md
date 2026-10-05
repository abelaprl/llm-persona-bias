# ETHICS — linguistic marker gabungan seluruh subset

Status: **kandidat, belum untuk klaim final tesis**. Run: 20261004_224737_740235.
Sumber pengukuran: D:\TugasAkhir\result\linguistic_markers\gender_ethics_raw_rebuild_20261004_213224_141860.
Inventaris: 81,756 respons; penjelasan tersedia: 65,711.
Tidak ada ekstraksi ulang, perubahan kamus, penilaian jawaban, atau API call.

## SPEC
Menambahkan ringkasan bahasa respons dari seluruh empat subset yang tersedia;
bukan seluruh populasi benchmark ETHICS dan bukan angka akurasi gabungan.
Semua enam model dan sembilan persona disertakan, tanpa filter committed/correct.
Data tanpa penjelasan tetap diinventarisasi, tetapi nilai marker tidak diganti nol.

## DESIGN PLAN
Satu respons tetap satu observasi. Pasangan berasal dari model, subset, ID tugas,
dan hash skenario yang sama. Subset asal tidak hilang saat penggabungan.
Rata-rata selisih berpasangan memakai bobot satu per pasangan lengkap: subset
dengan lebih banyak pasangan mendapat bobot lebih besar. Ini BUKAN bobot 25% per subset.
Bobot dapat berbeda antarmodel, persona, dan marker karena ketersediaan penjelasan.
Lihat subset_coverage_and_weights.csv dan paired_subset_weights.csv.

Rank-biserial dihitung ulang dari ranking seluruh selisih nonnol dalam gabungan,
bukan rata-rata rank-biserial per subset. Kontribusinya bergantung jumlah DAN
besar peringkat selisih; pooled_absolute_rank_share memperlihatkan kontribusi tersebut.
Prevalensi memakai respons dengan marker terukur; panjang penjelasan memakai
respons dengan jumlah kata tersedia. Penyebut setiap profil tersimpan dalam CSV.

## IMPLEMENTATION
Seluruh kode tambahan berada di notebook yang sama, bagian Gabungan seluruh subset.
18 PNG memakai warna penuh, arah Man − Woman, dan kolom NA terpisah.
Semua 41 marker masuk keluarga gambar: umum (termasuk penjelas ulang), etika,
gender, dan respons. Tidak ada marker baru atau klaim kesetaraan kamus dengan GSM8K.
Sumber dan pola kamus tetap pada registry.json / marker_registry_and_sources.csv.

Wilcoxon numerik dan McNemar exact untuk presence dihitung ulang dari pasangan;
sign test merupakan sensitivitas. Hasil Wilcoxon merupakan uji pada campuran
selisih gabungan, BUKAN uji efek yang dikontrol untuk subset; interpretasi lokasi
memerlukan simetri selisih. Pencampuran subset dapat melanggar asumsi ini.
Friedman numerik tambahan memakai kasus lengkap dan Holm antaroutcome per model/gender;
keluarganya terpisah dari uji pairwise, bukan konfirmasi independen.

Untuk p-value pairwise pada gambar gabungan, Holm mencakup 44 outcome × 18 kontras
× 5 tampilan (empat subset + gabungan) = 3.960 hipotesis per model.
Sign test dikoreksi dalam keluarga sensitivitas yang terpisah dengan ukuran sama.
Ini kebijakan tambahan setelah inspeksi, bukan preregistrasi.
joint_holm_subsets_and_pooled_candidate.csv menyediakan hasil koreksi bersama.
Gambar per subset LAMA tetap memakai keluarga awalnya (792 per model/subset);
jangan membacanya seolah sudah memakai keluarga gabungan yang baru.

CI 95% marginal untuk selisih RATA-RATA menggunakan 2000 bootstrap:
resampling pasangan soal di DALAM tiap subset, dengan jumlah pasangan per subset tetap.
CI ini bukan CI rank-biserial, bukan simultan, dan tidak mewakili ketidakpastian
akibat missingness atau validitas kamus. Median dan IQR delta disimpan terpisah.
Benih: 20261004. Tidak dilakukan imputasi.

## Membaca gambar
Negatif: Woman lebih tinggi; positif: Man lebih tinggi.
Warna penuh menunjukkan model, bukan signifikansi.
Lingkar hitam: p Holm < 0,05 (kandidat). Segitiga: kurang dari 20 pasangan nonnol.
Silang NA: rank-biserial tidak terdefinisi, dengan alasan di CSV.
Status no_change berarti semua pasangan sama: selisih mentah nol tetapi rank-biserial NA.
Status unavailable berarti tidak ada pasangan dengan ukuran tersedia.
Heatmap efek menunjukkan rank-biserial; tanda besar tidak otomatis didukung banyak pasangan.
Periksa n_pairs dan n_changed pada CSV, khususnya model dengan penjelasan sedikit.

## CHECKLIST / TESTING
- [x] Semua respons sumber masuk inventaris; source SHA-256 dicatat.
- [x] Tidak ada reasoning atau label acuan dalam pengukuran tambahan.
- [x] Pairing lintas subset dicegah dan kesamaan hash soal diperiksa.
- [x] NA, arah efek, bobot pasangan, bootstrap berstrata, dan Holm diperiksa.
- [x] 18 PNG lengkap, hash dan format PNG diverifikasi.
- [x] Gambar per subset dan report sumber tidak ditimpa.
- [ ] Audit manual precision/recall kamus dan validasi parsing untuk rilis final.

## Keterbatasan dan pertanyaan penguji
Ringkasan ini memudahkan membaca kecenderungan keseluruhan, tetapi dapat menyembunyikan
arah yang berlawanan antar-subset. Hasil gabungan tidak membuktikan efek seragam
di commonsense, deontology, justice, dan virtue. Rincian per subset tetap diperlukan.
Pertanyaan: apakah hasil didorong satu subset atau perbedaan ketersediaan penjelasan?
Jawaban diperiksa melalui bobot subset, coverage, dan hasil subset; jangan hanya
menunjukkan gambar gabungan. Equal-subset weighting belum dihitung.
Deontology/virtue dibaca sebagai bahasa terhadap prompt yang benar-benar terkirim,
bukan otomatis konstruk moral yang dimaksud benchmark asli.

## INTEGRATION
PNG tambahan bernama *_gabungan*.png di ETHICS/figures; versi subset tetap tersedia.
gallery_gabungan.html hanya berisi gambar gabungan; gallery.html menggabungkan keduanya.
Bab IV, Bab VI, GSM8K, data mentah, dan report lama tidak diubah.
Jalankan empat sel kode pada bagian ini untuk memakai ulang pengukuran aktif;
Run All membangun ulang sumber mentah dahulu, lalu membuat ringkasan gabungannya.
Setiap eksekusi membuat report baru; versi aktif yang tergantikan diarsipkan.
