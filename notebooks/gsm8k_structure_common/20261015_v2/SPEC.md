# SPEC
Periksa dan hitung ciri susunan yang masih tersimpan pada respons GSM8K gender, agama, dan regional. Seluruh split tersimpan, enam model, seluruh skenario; tanpa API, tanpa perubahan akurasi atau sumber lama. Sasaran: ukuran bersama yang tidak bergantung pergantian baris, dengan bukti audit respons utuh. Bukan rekonstruksi format asli yang hilang.

# DESIGN PLAN
1. Scan teks final/terlihat sumber beku ketiga domain; catat kandidat kata Step, nomor urut, bullet, dan Markdown heading/bold. Baca respons utuh sebelum menetapkan aturan.
2. Tetapkan aturan sempit untuk penanda struktur eksplisit yang masih terlihat, sama pada semua model/domain. Pisahkan frekuensi simbol/label dari jumlah langkah penalaran atau layout asli. Panjang kalimat tidak otomatis dipromosikan karena segmentasi matematika perlu pemeriksaan terpisah.
3. Audit setidaknya sepuluh respons per model dan 50 hit acak per aturan (semua jika hit kurang); simpan ledger audit agen, bukan validasi manusia independen. Periksa false positive operasi pengurangan, desimal, bold jawaban, nomor dalam isi soal. Jika gagal, perbaiki dan ambil sampel baru.
4. Verifikasi invariansi hasil terhadap perataan whitespace pada seluruh teks. Simpan keluaran kandidat baru, perbandingan dengan ukuran lama, seluruh provenance, dan galeri gambar.
5. Pengujian kandidat hanya setelah audit lulus: pasangan soal/model yang sama; effect size, delta, interval kepercayaan, Holm; keluarga tambahan terpisah dari hasil lama. Integrasikan hasil tervalidasi ke notebook sebagai tambahan yang jelas. Jangan mengganti hasil naskah secara diam-diam.

# IMPLEMENTATION
Skrip lokal di folder kerja ini; instalasi modul dan hasil baru di repo eksperimen setelah validasi.

# CHECKLIST
- Seluruh respons; nol dibedakan dari teks tidak tersedia.
- Tidak membaca ground truth untuk menetapkan ukuran.
- Aturan dan satuan sama pada ketiga domain; hasil lama utuh.
- Semua klaim mengikuti bukti respons tersimpan.

# TESTING
Audit positif/negatif, invariansi whitespace seluruh korpus, tidak ada duplikasi pasangan, hash sumber tidak berubah, statistik keluarga dapat direproduksi.

# INTEGRATION
Tambahan notebook beserta status audit dan gambar. Ringkasan kepada pengguna menyebut tepat ukuran yang dapat dipakai tanpa rerun model. Naskah hanya diperbarui dengan hasil tervalidasi bila lingkup penerapan sudah jelas.
