# Audit precision kandidat

Baca `precision_full_blind.json` bersama definisi marker di registry. `visible_start/end` memakai offset karakter Unicode pada teks final yang ditampilkan; span akhir eksklusif. Baca seluruh respons sebelum menilai span. Isi correct_match dengan 0/1, identitas reviewer sebenarnya, full_response_read, tipe kesalahan dan alasan. Blank bukan nol.

Reviewer A dan B bekerja independen sebelum adjudikasi. `precision_design_key.csv` menyimpan provenance model/persona/rule/raw span untuk evaluasi; jangan memakai identitas model atau ground truth jawaban untuk memilih label marker. Paket precision tidak menggantikan recall tanpa anchor di paket held-out terpisah. Seluruh berkas saat dibuat adalah template kosong, bukan gold.
