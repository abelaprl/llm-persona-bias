# SPEC
Lengkapi pengukuran nilai dan norma budaya pada GSM8K regional-kultural dengan sembilan aturan yang sudah dipakai ETHICS. Gunakan seluruh respons tersimpan enam model dan 17 skenario; ETHICS mencakup empat subset dan gabungan. Tidak menjalankan inferensi atau mengubah penilaian jawaban.

# DESIGN PLAN
Registry beku: regional_ethics_raw_rebuild_20261005_231140_525498. Input GSM8K: regional_gsm8k_raw_rebuild_20261005_231108_470909. Terapkan sembilan marker berakhiran _orientation pada analysis_text yang sudah melalui pemisahan teks terlihat dan jawaban teknis; gunakan word_count dan explanation_available sumber yang sama. Verifikasi hasil ETHICS mereproduksi count/density sebelumnya. Hitungan mentah dan kemunculan disimpan, tetapi pengujian hanya frekuensi per 100 kata sesuai baris metode.

Pasangkan subset + question_id + question_hash, menggunakan pair_eligible. Satu keluarga tambahan per model/tugas: sembilan frekuensi x 52 perbandingan; ETHICS x lima cakupan. Wilcoxon pendekatan normal, rank-biserial, delta, paired bootstrap 2000, Holm (468/2340). Hipotesis tanpa pasangan tetap dalam ukuran keluarga. Jangan menyatukan nilai budaya menjadi satu skor atau menyimpulkan nilai tokoh soal sebagai orientasi model.

# IMPLEMENTATION
Modul mandiri memakai pencocokan dan penyaring konteks yang sudah ada; aturan tidak diperluas berdasarkan signifikansi. Output baru terpisah, notebook regional mendapat bagian pengukuran bersama.

# CHECKLIST
Audit teks respons lengkap untuk temuan GSM8K, periksa kata umum/isi soal, simpan span dan sumber. Label tabel hanya diperluas setelah pengukuran lengkap dan audit selesai. Hasil lama dipertahankan.

# TESTING
Reproduksi ukuran ETHICS pada semua respons; periksa jumlah keluarga, kunci pasangan, missing versus nol, ketidakterlibatan ground truth, dan kecocokan angka naskah dengan CSV.

# INTEGRATION
Tabel IV.9, keluarga Holm IV.11, satu uraian ringkas hasil di VI, bagian notebook GSM8K/ETHICS; compile dan periksa PDF. Tidak commit/push.
