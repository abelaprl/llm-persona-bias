# Kontras GSM8K ditahan

Pada Llama 3.1, `gsm8k_00511` adalah soal Gerald pada Neutral tetapi soal Andy pada E dan I1. Bukti teks, path JSON, dan SHA-256 pertanyaan tersedia di [pairing_counterexample.csv](tables/pairing_counterexample.csv). 

Modul statistik GSM8K lama di `src/MAIN_PARSING/Linguistic Marker Builder/` (`religion_statistics.py` dan `regional_statistics.py`, baris 95–107) memasangkan hanya berdasarkan question_id. Keunikan ID dan hash file tidak membuktikan kesamaan soal. Satu counterexample ini membatalkan klaim bahwa seluruh pairing sudah valid; **bukan** bukti bahwa setiap model atau setiap pasangan salah.

Seluruh U04/U05 GSM8K (E dan I1–I3) ditahan secara konservatif hingga audit pairing berdasarkan isi soal dilakukan. Estimate/CI display = NA, bukan nol. Kolom source_* dan statistik asli hanya arsip provenance, tidak boleh dipakai untuk kesimpulan. Grafik deskriptif tetap mengikuti populasi sumber. U07 dipilih terpisah sebagai contoh yang diperiksa. Statistik ETHICS tidak diubah.

Perbaikan memerlukan audit isi soal seluruh pasangan lalu penghitungan ulang statistik dan Holm pada pasangan yang sah. Langkah tersebut belum dilakukan dalam tugas penyajian ini.
