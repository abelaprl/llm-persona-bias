# Mulai meninjau gambar penanda linguistik gender — Bab VI v3

Pembaruan caption karakteristik dan zoom jumlah kata ETHICS tersedia di
[panduan v3.1](MULAI_DI_SINI_BAB6_V31_20261007.md). Heatmap di bawah tetap berlaku.

Versi 7 Oktober 2026. Presentasi baru memakai statistik dukungan v2 yang
dibekukan. Status kamus tetap kandidat. Seluruh enam model dan variasi
I1–I3 tersedia, tanpa pemilihan berdasarkan signifikansi.

## Mulai dari pratinjau halaman

Buka PDF pada ukuran sebenarnya atau cetak 100%, tanpa fit-to-page.
Gambar selebar 14 cm berada di bidang teks A4 sesuai margin tesis.

- [GSM8K: tiga halaman contoh A4](../../../result/analyses/linguistic_markers/runs/gender_gsm8k_bab6_v3_20261007_230650_742192/pratinjau_A4_ukuran_asli.pdf)
- [ETHICS: empat halaman contoh A4](../../../result/analyses/linguistic_markers/runs/gender_ethics_bab6_v3_20261007_230725_835481/pratinjau_A4_ukuran_asli.pdf)

## Contoh PNG untuk diperiksa

| Perbandingan | GSM8K | ETHICS |
|---|---|---|
| Man terhadap N: kepastian dan keraguan | [PNG](GSM8K/figures/bab6_v3_20261007_230650_742192/gsm8k__01_man_vs_neutral__02_kepastian.png) | [Commonsense](ETHICS/figures/bab6_v3_20261007_230725_835481/commonsense__01_man_vs_neutral__02_kepastian.png) |
| Woman I−E: kepastian dan keraguan | [PNG](GSM8K/figures/bab6_v3_20261007_230650_742192/gsm8k__04_woman_vs_explicit__02_kepastian.png) | [Commonsense](ETHICS/figures/bab6_v3_20261007_230725_835481/commonsense__04_woman_vs_explicit__02_kepastian.png) |
| Selisih rata-rata jumlah kata: Man terhadap N | [PNG](GSM8K/figures/bab6_v3_20261007_230650_742192/gsm8k__01_man_vs_neutral__00_word_count.png) | [Commonsense](ETHICS/figures/bab6_v3_20261007_230725_835481/commonsense__01_man_vs_neutral__00_word_count.png) |
| Dukungan rendah, TP, ND | — | [Konteks Justice](ETHICS/figures/bab6_v3_20261007_230725_835481/justice__01_man_vs_neutral__05_konteks.png) |

Contoh dipilih untuk memeriksa jenis tampilan dan dukungan pasangan,
bukan karena signifikan. PDF vektor setiap gambar memiliki nama yang sama
dengan PNG dan ekstensi `.pdf`.

## Paket lengkap dan tabel

- [50 pasangan PNG/PDF GSM8K](GSM8K/figures/bab6_v3_20261007_230650_742192/)
- [275 pasangan PNG/PDF ETHICS](ETHICS/figures/bab6_v3_20261007_230725_835481/)
- [Tabel satuan asli GSM8K: 2.268 perbandingan](../../../result/analyses/linguistic_markers/runs/gender_gsm8k_bab6_v3_20261007_230650_742192/all_comparisons_native_units.csv)
- [Tabel satuan asli ETHICS: 23.760 perbandingan](../../../result/analyses/linguistic_markers/runs/gender_ethics_bab6_v3_20261007_230725_835481/all_comparisons_native_units.csv)
- [Laporan perubahan dan verifikasi](../../../docs/gender_lm_bab6_ringkas_20261007.md)

Setiap paket memiliki `figure_index.csv` dengan caption dan prioritas
utama/pendamping. Urutan utama: persona−N dan I−E. Man−Woman dan gabungan
ETHICS menjadi pendamping. I1–I3 adalah variasi isi warmup, bukan tingkatan.

Inti Bab IV mencakup 16 ukuran GSM8K dan 21 ETHICS, termasuk tiga
karakteristik teks. Penjelas ulang dipisahkan dari inti. Pengelompokan
lengkap dan sumber teori ada pada `marker_chapter_iv_mapping.csv` di run.
Indeks `appendix_v2_index.csv` menunjuk gambar v2 lengkap yang tetap utuh;
gambar historis itu mempertahankan notasi lamanya. Tabel dan gambar baru
memakai k/n (berubah/tersedia) secara konsisten.

Font dukungan terkecil 7 pt. Contoh halaman diperiksa pada layar; uji cetak
fisik belum dilakukan. Hasil baru belum disinkronkan ke Bab IV/VI.
