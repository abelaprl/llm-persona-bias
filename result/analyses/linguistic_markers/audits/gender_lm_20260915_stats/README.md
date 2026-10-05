# Audit statistik Gender ETHICS — 15 September 2026

Audit ini membaca angka keluaran aktif dan menghitung ulang statistik dari tabel respons beku; tidak mengubah kode produksi, label jawaban, anotasi audit manusia, atau keluaran lama. Seluruh angka berikut berstatus *candidate* mengikuti sumbernya.

Sumber: `result/ETHICS/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_20260909_r4/`. Perintah reproduksi audit: `.venv/Scripts/python.exe src/MAIN_PARSING/ETHICS/audit_gender_lm_statistics_20260915.py --output-dir result/audit_gender_lm_20260915_stats_rerun`. Direktori tujuan harus baru; skrip menolak direktori yang sudah ada. `recompute.py` adalah salinan skrip yang menjalankan audit awal.

## Temuan 1: uji utama berbeda dari Bab IV

`D:/ProposalTA-18222008/Bab IV - Perancangan.tex:1377–1382` dan `:1484–1485` menetapkan Wilcoxon signed-rank dua sisi, sedangkan uji tanda menjadi pemeriksaan tambahan untuk banyak selisih nol. `src/MAIN_PARSING/Linguistic Marker Builder/statistics.py:151–157` menjadikan uji tanda sebagai `p_raw` ketika proporsi nol >50% atau |Bowley skew| >0,5. Ini perbedaan spesifikasi analisis, bukan kesalahan hitung p-value.

Pada 2.700 uji numerik/densitas *all_text*: 1.585 memakai uji tanda, 579 Wilcoxon, dan 536 `no_change`. Sensitivitas berikut mengganti uji utama dengan Wilcoxon, mempertahankan keluarga Holm serta perlakuan `no_change` p=1; hasilnya bukan pengganti hasil aktif.

| Keluaran | Populasi | Uji | Signifikan aktif | Signifikan Wilcoxon | Hilang | Muncul |
|---|---|---:|---:|---:|---:|---:|
| Numerik umum | all_text | 2.268 | 776 | 750 | 79 | 53 |
| Densitas gender | all_text | 432 | 114 | 111 | 7 | 4 |
| Numerik umum | committed | 2.268 | 553 | 550 | 56 | 53 |
| Densitas gender | committed | 432 | 83 | 86 | 1 | 4 |

Ada 143/2.700 keputusan signifikansi *all_text* yang berubah: 86 tidak lagi signifikan dan 57 menjadi signifikan. Ketentuan uji utama perlu diselaraskan antara naskah dan implementasi sebelum klaim inferensial difinalkan. Pilihan tidak boleh ditentukan dari hasil yang lebih signifikan.

## Temuan 2: arah warna dan dasar bintang dapat menunjuk aspek berbeda

`sign_rank_direction_disagreement.csv` memuat 116 uji *all_text* yang menggunakan uji tanda tetapi arah `rank_biserial` berlawanan dengan `n_positive - n_negative`. Empat signifikan Holm; lihat `significant_direction_disagreement.csv`. Ini dua estimand berbeda: uji tanda menghitung arah pasangan, sedangkan rank-biserial memberi bobot berdasarkan peringkat besar selisih. Keduanya dapat sah secara aritmetis sekaligus memberi arah berbeda.

Contoh konkret pada `numeric_vs_neutral_all_text.csv:907`: Qwen3, persona man, contextual hedge I1−N mempunyai 151 pasangan positif dan 77 negatif, tetapi rank-biserial −0,03723; p Holm dari uji tanda sekitar 0,000004. Pada `numeric_vs_explicit_all_text.csv:569`: GPT-OSS, persona woman, word_count I1−E mempunyai 111 positif dan 201 negatif, tetapi rank-biserial +0,06056; p Holm sekitar 0,000002, sedangkan p Wilcoxon 0,35105.

Renderer `src/MAIN_PARSING/ETHICS/present_ethics_lm_overall.py:182–193` menggunakan warna efek dan bintang `p_holm`, dengan caption rank-biserial pada `:196`. Pembaca tidak boleh mengartikan bintang otomatis sebagai dukungan arah perubahan warna. Jika uji tanda tetap digunakan sebagai analisis utama, arah serta ukuran efek yang sesuai perlu diperlihatkan atau perbedaan kedua estimand dijelaskan.

## Dukungan pasangan dan batas interpretasi besarnya efek

Pada 2.700 uji *all_text*, 342 mempunyai |rank-biserial| ≥0,8 meskipun ≥95% selisih nol. Sebanyak 117 uji memiliki rank-biserial ±1 dengan hanya satu pasangan berubah. Contoh `numeric_vs_neutral_all_text.csv:20`: DeepSeek, organization_bullet, I2−N, 1.488 pasangan tetapi hanya satu berbeda, r=1 dan p Holm=1. Rumus pada `statistics.py:111–118` memang membuang selisih nol sebelum memberi peringkat. Nilai ekstrem itu bukan bukti perubahan luas pada seluruh respons; tampilkan `n_nonzero`/`n_pairs` ketika membahas besarnya efek.

## Pemeriksaan dan batas audit

Penghitungan Holm independen dari seluruh tabel numerik/densitas cocok dengan hasil tersimpan hingga galat pembulatan <7×10⁻¹⁶. Keluarga yang digunakan ialah tujuh kontras per model×identity×outcome dan empat kontras langsung per model×outcome, sesuai manifest; ini tidak membuktikan kontrol galat secara global lintas seluruh marker/model.

`recompute.py` selesai dengan exit code 0 dan menghitung ulang 7.776 uji numerik, densitas, dan presence pada populasi *all_text* serta *committed*. Tabel sumber mempunyai 81.756 baris dan 81.756 kunci unik. Pasangan dibentuk berdasarkan `(subset, question_id)` secara independen dari fungsi pooling produksi: tidak ada jumlah pasangan yang berbeda pada 7.776 uji. Galat absolut maksimum p-value mentah, Holm, dan ukuran efek masing-masing tidak melebihi 1,12×10⁻¹⁶. Angka sensitivitas di atas juga terkonfirmasi dari tabel respons. Hasil lengkap tersedia pada `summary.json`, `summary.csv`, dan `independent_recomputation.csv`; CSV audit menyimpan nama file dan nomor baris sumber. SHA-256 tabel respons cocok dengan manifest aktif: `c816e8909a8152291dd6edcbbea90f170b3f458eb053242cd2ef18b817c6de77`.

Audit ini tidak memvalidasi makna hit marker, tidak menilai benar/salah jawaban, tidak menghitung ulang interval bootstrap atau uji pembuka multinomial, dan tidak mengesahkan gerbang audit manusia. Pasangan *committed* diperiksa hanya sebagai sensitivitas; analisis utama tetap mencakup seluruh respons dengan teks tersedia.
