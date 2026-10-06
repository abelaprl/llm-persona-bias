# Paket anotasi manusia

Status: belum dianotasi. Folder reviewer_a/reviewer_b adalah tempat kerja, bukan identitas reviewer nyata.

1. Tentukan dua reviewer manusia. Keduanya membaca full_final_blind.json secara independen.
2. Baca codebook.md. Tandai semua kategori per respons (present=0/1); jangan mengisi blank otomatis sebagai nol.
3. Catat span dan konteks. Posisi surface mengacu ke teks yang ditampilkan; raw offset harus dipetakan ke final sumber sebelum exact-span evaluation.
4. Isi completion.csv untuk setiap respons, termasuk no-hit atau label-only.
5. Hitung agreement sebelum berbagi jawaban; adjudikasi disimpan terpisah setelahnya.
6. Sampling key dipakai evaluator untuk cluster pertanyaan dan inclusion probability, bukan sebagai label.

Jangan menyebut paket ini gold sebelum dianotasi dan provenance diperiksa. Precision per aturan membutuhkan paket hit dari run keseluruhan, bukan hanya paket recall ini.
