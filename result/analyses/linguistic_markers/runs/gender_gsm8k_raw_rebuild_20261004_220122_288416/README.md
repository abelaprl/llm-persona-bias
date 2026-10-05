# GSM8K: hitung ulang dari JSON asli

Status: **kandidat, belum untuk klaim final tesis**. Run: 20261004_220122_288416.
Notebook: linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb. Seluruh kode perbaikan berada di notebook.
Data mentah, report lama di result/, dan Bab IV/VI tidak diubah.

## SPEC
Seluruh JSON yang tersedia untuk enam model dan sembilan persona, bukan sampel.
Jumlah JSON: 71,226; penjelasan tersedia: 69,391.
Reasoning tanpa content: 781; tidak dipakai sebagai penjelasan.
Seluruh respons diinventarisasi, tanpa filter committed/correct.

## DESIGN PLAN
GSM8K dicocokkan dengan teks soal pada seluruh 1.319 soal test lokal.
ETHICS dipasangkan menggunakan ID tugas dan hash skenario; empat subset tidak dilebur.
Nomor file lama dipertahankan untuk audit, bukan dianggap bukti soal identik.
Duplikasi dan soal yang tidak dapat dicocokkan tetap dalam inventaris, dikeluarkan dari pairing.

## IMPLEMENTATION
Kanal message.content menjadi sumber utama, termasuk bila kosong.
Tidak ada fallback ke reasoning atau prediction.raw. Wrapper content hanya dipakai ketika
field message.content tidak tersedia dan sumbernya dicatat.
Kamus marker lama dipertahankan, sumber/definisi pada marker_registry_and_sources.csv.
Respons yang seluruhnya bilangan bulat ASCII dicatat sebagai answer_only_numeric,
bukan penjelasan satu kata. Dasar: 102 respons utuh GPT-OSS telah diperiksa;
aturan berlaku sama untuk semua model/persona, tanpa menilai kebenaran jawaban.
Snapshot dapat dipakai melalui LM_GSM8K_SNAPSHOT; semua hash JSON, kamus,
dan modul diperiksa dahulu. Tanpa variabel itu, notebook membaca ulang seluruh JSON.
Kamus ETHICS dan GSM8K tidak diklaim identik; penamaan serupa tidak membuktikan ekuivalensi aturan.
Density per 100 kata; count dan presence tersedia per respons.
NA tidak diganti nol. Semua pasangan bernilai sama menghasilkan selisih mentah nol,
tetapi rank-biserial tanpa pasangan nonnol tidak terdefinisi (NA; status no_change).
NA juga dapat berarti tidak ada pasangan tersedia (status unavailable); alasan ada di CSV.

## CHECKLIST
- [x] Seleksi kanal, tidak memakai reasoning.
- [x] Verifikasi isi soal setiap pasangan dan penolakan duplikasi.
- [x] Uji arah Man minus Woman, batas rank-biserial, NA dan koreksi Holm.
- [x] Hasil baru di path baru; PNG lama dapat dipulihkan dari arsip.
- [ ] Audit manual precision/recall marker.
- [ ] Validasi lengkap parsing jawaban dan lineage untuk sensitivitas correctness.
- [ ] Review konstruk distancing, empati, serta overlap kontras/konsesif.
- [ ] Rilis final untuk tesis.

## TESTING
Jalankan seluruh sel dari kernel baru. Assertions menggagalkan instalasi bila invariant gagal.
Notebook menghasilkan folder run baru setiap eksekusi, tanpa menimpa report di result/.
source_audit.csv dan pairing_audit.csv memuat bukti; source JSON memiliki SHA-256.
source_hashes pada manifest merekam versi modul; salinan kode tersimpan di code_snapshot/.

## Membaca gambar dan statistik
Semua titik memakai warna penuh yang sama untuk enam model.
Man minus Woman: negatif berarti Woman lebih tinggi, positif berarti Man lebih tinggi.
Tanda silang di kolom NA: tidak ada estimasi; bukan nol.
Segitiga: kurang dari 20 pasangan dengan selisih nonnol, **bukan** otomatis hasil tak sah.
Lingkar hitam: p Holm < 0,05 pada statistik kandidat; bukan bukti kamus sudah valid.
Keluarga Holm: seluruh outcome x 18 kontras per model dan subset, ditetapkan untuk rebuild ini;
ini perubahan analisis setelah inspeksi, bukan preregistrasi.
Wilcoxon selalu menjadi uji numerik utama; sign test menjadi sensitivitas, tidak fallback.
Presence menggunakan McNemar exact. Friedman numerik memakai soal lengkap N/E/I1/I2/I3.
Friedman tidak dihitung untuk outcome biner.
CI 95% adalah selisih rata-rata berpasangan dengan 2000 bootstrap,
bukan CI rank-biserial; median delta dan IQR disediakan terpisah di CSV.
Tidak ada ambang besar/kecil otomatis untuk efek yang bertumpu sedikit pasangan.
TTR memakai hingga 300 kata alfabetis pertama; panjang <300 dicakup dan dapat memengaruhi TTR.
Grafik correctness lama tidak dipakai ulang karena lineage jawaban belum tervalidasi.

## INTEGRATION
Hasil hanya dipasang ke notebook dan folder figures benchmark; tidak disalin ke ProposalTA.
Grafik lama diarsipkan untuk mencegah hasil lama bercampur dengan rebuild.
Temuan ETHICS deontology/virtue dibaca sebagai bahasa respons terhadap prompt yang terkirim,
bukan otomatis pengukuran kemampuan moral sesuai benchmark asli.
