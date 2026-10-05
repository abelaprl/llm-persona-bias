# GSM8K: visual tambahan dengan dukungan pasangan — v2

Run: 20261004_232525_357456. Status: **kandidat; bukan rilis final tesis**.
Visual dan metadata lama tetap utuh; tidak ada file lama dihapus, dipindahkan, atau ditimpa oleh bagian tambahan ini.
Implementasi hanya berada di bagian Dukungan pasangan v2 pada notebook benchmark yang sama.
Tidak ada sinkronisasi ke Bab IV/VI, perubahan kamus, scoring, API call, atau filter committed/correct.

## SPEC
Seluruh 71,226 respons pada hasil pengukuran aktif, tanpa sampling tambahan.
Sumber per respons: D:\TugasAkhir\result\linguistic_markers\gender_gsm8k_raw_rebuild_20261004_220122_288416.
Sumber statistik gabungan: tidak berlaku.
Jumlah mentah marker tetap tersedia dalam response_records.csv.gz sumber.
Bagian ini memakai density per 100 kata atau presence sesuai registry yang dibekukan.

## DESIGN PLAN
Satu pasangan harus memiliki model, subset, ID tugas, dan hash soal yang sama.
Seluruh 2,268 perbandingan diperiksa ulang jumlah pasangan, jumlah perubahan,
selisih rata-rata, dan rank-biserialnya terhadap pengukuran per respons.
I1–E, I2–E, I3–E merupakan perbandingan langsung. Tidak dihitung dengan mengurangkan
rank-biserial persona–N. Semua 18 kontras tercakup, termasuk antarpersona dan terhadap N.
Ukuran efek, jumlah pasangan, dan CI tersimpan tidak diubah; p-value kandidat ditinjau terpisah.

## IMPLEMENTATION / statistik
Numerik dengan 1–20 selisih nonnol: signed-rank eksak bersyarat melalui semua kemungkinan
tanda pada peringkat absolut; dynamic programming menghitung distribusinya tanpa Monte Carlo.
Ties memakai average ranks, selisih nol dikeluarkan, arah dua sisi. Batas 20 adalah pilihan
cabang komputasi, BUKAN batas validitas atau ukuran efek. Numerik dengan >20 selisih nonnol
mempertahankan Wilcoxon asimtotik sumber. Presence mempertahankan McNemar exact.
P=1 ketika semua selisih nol merupakan konvensi pelaporan no_change, bukan bukti kesetaraan populasi.
Eksak tidak menghilangkan asumsi simetri/pertukaran tanda, independensi pasangan, atau bias seleksi teks.

Holm untuk semua gambar v2 memakai 378 hipotesis per model:
21 outcome × 18 kontras pada GSM8K.
Sign test tetap sensitivitas, dengan keluarga Holm terpisah berukuran sama.
Ini perubahan analisis setelah inspeksi, bukan preregistrasi. Jangan memilih versi lama/baru
berdasarkan signifikansi. previous_vs_support_v2.csv menyimpan p lama–baru dan ukuran keluarga.
Gambar lama tidak mendapat koreksi v2 secara diam-diam; gunakan satu rangkaian v2 secara konsisten.

## Membaca heatmap
Angka utama: matched rank-biserial, dihitung dari selisih nonnol.
Angka k/n: k pasangan berubah dari n pasangan tersedia, bukan ukuran confidence interval.
[d]: kejadian per 100 kata; [b]: hadir/tidak pada respons, bukan density.
Warna pekat berarti rank-biserial mendekati ±1, bukan berarti banyak respons berubah.
Arsir: 1–19 pasangan berubah, aturan penandaan kehati-hatian. Flag low_support CSV
menggunakan n_changed < 20 secara konsisten, termasuk no_change yang ditampilkan terpisah.
TP: pasangan tersedia tetapi tidak ada perubahan teramati; rank-biserial tetap NA, tidak dijadikan nol.
ND: tidak ada pasangan untuk marker tersebut. Keduanya dibedakan teks DAN warna latar.
Bintang: p Holm < 0,05. Tidak ada bintang bukan bukti kesetaraan; bintang bukan bukti efek besar.
Angka unit count/density/presence dan status lengkap berada pada CSV.
Garis pada keluarga etika memisahkan penanda konteks/afek dari framing/evaluasi;
tidak menyatakan salah satu kelompok sudah lebih tervalidasi.
Label sikap jawaban tegas adalah marker bahasa, bukan status committed milik parser jawaban.

## Cakupan dan CI
Cakupan pasangan memakai penjelasan tersedia di KEDUA skenario dibanding jumlah soal
yang dikirim per skenario. Cakupan respons lama tetap diperlukan sebagai pendamping.
Grafik karakteristik memakai selisih rata-rata, bukan rank-biserial.
CI bootstrap 95% marginal diwarisi dari sumber; n=1 disembunyikan pada gambar tetapi
ci_low/ci_high asli tetap tersimpan. n=0 tidak diberi estimasi. n=2 atau lebih bukan otomatis cukup:
n<20 diberi penanda kehati-hatian; jumlah pasangan selalu dicantumkan.
Skala tiap ukuran disamakan antarskenario dalam gambar; satuan jumlah kata, panjang kalimat,
dan TTR tidak dicampur. CI tidak mewakili ketidakpastian akibat respons tanpa penjelasan.
Grafik CI memakai n/k (tersedia/berubah), sedangkan heatmap memakai k/n; legenda masing-masing eksplisit.

## Gabungan ETHICS
Jika ada gabungan, bobotnya mengikuti pasangan tersedia, bukan bobot 25% per subset.
Rank-biserial diranking ulang pada selisih gabungan; bukan rata-rata rank-biserial subset.
Hasil gabungan tidak membuktikan arah seragam. Periksa bobot subset dari report gabungan sumber
dan hasil per subset. Deontology/virtue dibaca sebagai bahasa terhadap prompt yang benar-benar dikirim,
bukan otomatis konstruk benchmark asli atau akurasi gabungan.

## CHECKLIST / TESTING
- [x] 23 pemeriksaan eksak, ties, nol, batas k=20, dan Holm.
- [x] Rekonstruksi seluruh pasangan/efek cocok dengan tabel sumber.
- [x] Semua marker tercakup tepat sekali dalam keluarga visual.
- [x] Pemisahan TP/ND, suppression CI n=1, dan definisi low_support konsisten.
- [x] 12 PNG tambahan diverifikasi; berkas lama diperiksa hash-nya.
- [ ] Audit manual precision/recall kamus dan validasi jawaban untuk rilis final tesis.
- [ ] Penyesuaian caption dan pemilihan gambar Bab VI; tidak dilakukan pada tugas ini.

## INTEGRATION
Jalankan hanya empat sel pada bagian Dukungan pasangan v2 untuk menghasilkan versi tambahan.
Tidak perlu Run All: bagian legacy di atasnya memiliki workflow rebuild dan arsip tersendiri.
Setiap eksekusi tambahan menghasilkan nama bertanggal/waktu baru, tanpa mengganti versi lama.
Galeri baru mengelompokkan subset dan jenis perbandingan. Caption draf ada pada figure_index.csv.
Galeri lama tetap utuh, sehingga temuan kandidat lama tidak tercampur dengan metode v2.

## Rujukan metode
- Wilcoxon dan perlakuan ties/zeros: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- Permutasi tanda berpasangan: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- Holm: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html
Kamus marker dan sumber teorinya tetap mengikuti registry.json sumber, bukan literatur baru yang diklaim telah diimplementasikan.
