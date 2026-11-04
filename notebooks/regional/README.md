# Notebook Regional Cultural yang dipakai

Notebook utama sekarang memakai **Regional v3 r5** dengan gaya visual setara
Religion: 53 marker, termasuk seluruh 23 kategori regional, kartu respons,
keragaman kategori, overlap literal, serta statistik berpasangan baru.
[Bundle terverifikasi](../../result/benchmarks/ethics/reports/regional_ethics_presentation_20260919_r2/verification.json)
memuat 77 PNG dan 77 SVG. Status tetap **candidate**, belum lolos audit dua
reviewer manusia. [Catatan pelaksanaan](../../docs/regional_ethics_presentation_execution.md)
menjelaskan denominator, pengecualian pairing, verifikasi dan keterbatasan.

| Kebutuhan | Buka |
|---|---|
| Linguistic marker ETHICS utama | [Notebook v3](<linguistic marker/ETHICS/linguistic_marker_regional_cultural_ethics_v3_candidate.ipynb>) |
| Semua gambar utama | [Figures PNG/SVG](<linguistic marker/ETHICS/figures/>) |
| Teks utuh dan sorotan marker | [Kartu respons HTML](../../result/benchmarks/ethics/reports/regional_ethics_presentation_20260919_r2/evidence/response_cards.html) |
| Notebook tesis historis | [Notebook tesis v2.0.1](linguistic_marker_regional_cultural_ethics_thesis_v2_0_1_candidate_20260828.ipynb) |
| Atlas historis | [Atlas v2.0.1](linguistic_marker_regional_cultural_ethics_atlas_v2_0_1_candidate_20260828.ipynb) |
| Akurasi ETHICS | [Notebook akurasi aktif](<ethics accuracy analysis/ETHICS_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb>) |
| Akurasi GSM8K | [Notebook akurasi](<akurasi analysis/GSM8K_Regional_Cultural_Accuracy_Analysis_Lengkap.ipynb>) |

Notebook v3 mempertahankan seluruh 154.428 respons. Analisis bahasa memakai
126.723 respons berpenjelasan, termasuk abstain; label-only bukan nol marker
berpenjelasan. Sebanyak 20 respons DeepSeek neutral dengan soal berbeda
dikecualikan **hanya dari pairing**, bukan dihapus dari coverage/deskriptif.
[Daftar pengecualian](../../result/benchmarks/ethics/reports/regional_ethics_presentation_20260919_r2/tables/pairing_exclusions.csv)
dan [tabel coverage](../../result/benchmarks/ethics/reports/regional_ethics_presentation_20260919_r2/tables/coverage.csv)
menyimpan denominator tersebut. Seluruh gambar aktif langsung berada dalam
satu folder `linguistic marker/ETHICS/figures/`, tanpa subfolder versi.

## Hasil historis yang dipertahankan

Notebook tesis v2 memakai statistik gabungan dari respons beku v2.0.1. Atlas tetap
menyimpan rincian sumber per subset dari `regional_lm_v2_0_1_complete_20260828`, sesuai
[catatan eksekusi proyek](../../docs/regional_cultural_ethics_linguistic_markers_v2_execution.md).
Status ilmiahnya tetap **candidate**, belum hasil final tesis.

Tiga puluh gambar versi 13 September tetap ada di
[figures keseluruhan ETHICS historis](../../result/benchmarks/ethics/reports/overall_presentation_20260913_r2/regional/figures/).
[Penjelajah tabel historis](../../result/benchmarks/ethics/reports/overall_presentation_20260913_r2/regional/explorer.html)
dan [catatan audit historis](../../docs/ethics_lm_overall_presentation_20260913.md)
tetap dapat dibuka; p-value lama tidak dipakai untuk denominator baru.
Gambar sebelumnya tetap ada di [figures sumber](../../result/benchmarks/ethics/reports/Regional_Cultural_Persona/linguistic_markers/analysis/regional_lm_v2_0_1_complete_20260828/figures/).
Gambar akurasi aktif ada di `ethics accuracy analysis/figures/`.

Empat notebook lama dan satu folder render gagal dipindahkan ke
[arsip Regional ETHICS](../archive/20260910_regional_ethics/).
`archive_index.csv` mencatat lokasi asal dan SHA256 untuk pemulihan.
Data/gambar sumber di `result/` serta notebook GSM8K tetap di lokasi semula.
