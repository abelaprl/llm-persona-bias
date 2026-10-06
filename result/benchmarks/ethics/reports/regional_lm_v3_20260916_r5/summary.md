# Ringkasan kandidat Regional/Cultural ETHICS v3

Status rilis: **candidate** (`release_eligible=false`). Dataset memuat 154,428 respons. Analisis jawaban memakai hasil parser program yang sudah ada; baris `semantic_unaudited` didemote menjadi *abstain* hanya pada view analisis.

Perubahan marker teramati pada 2 sel model–persona–subset–marker. Perubahan aturan terbatas pada pengecualian idiom `if you will` dari `connective_conditional`. Statistik 55 outcome yang tidak berubah dipakai ulang setelah kesetaraan seluruh kolom marker/surface dibuktikan; outcome conditional dihitung ulang dengan 10.000 bootstrap.

Akurasi hanya ditafsirkan untuk commonsense dan justice. Deontology dan virtue disajikan sebagai distribusi jawaban. Uji status/availability terhadap persona adalah diagnosis asosiasi dan bukan estimasi error detector. Perbedaan commitment terdeteksi pada 24/24 uji Holm dan perbedaan linguistic availability pada 12/24 uji Holm. Hasil ini menunjukkan risiko coverage yang tidak seragam; hasil tersebut tidak mengukur FP/FN detector. Uji error detector terhadap persona tidak tersedia tanpa gold manusia valid.

Readback AI diagnostik mencakup 18 respons, 166 hit diterima, dan label {'TP': 157, 'FP': 6, 'UNCERTAIN': 3}; cakupan rule 34/122, dengan 9 catatan kandidat FN. Ini data development yang pernah terekspos, bukan estimasi precision/recall independen atau pengganti reviewer manusia.

Gate tetap tertutup sampai dua reviewer manusia independen dan adjudikasi selesai.
