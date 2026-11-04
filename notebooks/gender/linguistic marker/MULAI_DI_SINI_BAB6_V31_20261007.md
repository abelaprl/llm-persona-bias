# Pemeriksaan PNG dan caption Bab VI — v3.1

Versi 7 Oktober 2026. Perubahan terbatas pada caption karakteristik teks
dan tambahan panel pembesaran jumlah kata ETHICS. Heatmap v3 beserta judul
Man/Woman tetap digunakan. Statistik dan status kamus kandidat tidak berubah.

## Mulai dari dua gambar ini

- [ETHICS Commonsense, Man − Woman: lengkap dan pembesaran 5×](ETHICS/figures/bab6_v3_1_20261007_234752_514764/commonsense__05_direct__00_word_count_zoom.png).
- [GSM8K Man terhadap N: caption karakteristik baru](GSM8K/figures/bab6_v3_1_20261007_234710_219408/gsm8k__01_man_vs_neutral__00_word_count.png).

Panel lengkap di kiri mempertahankan semua CI. Panel kanan memperbesar
rentang sekitar nol sebanyak 5×. Ujung panah menandai CI yang melewati
batas panel; segitiga terbuka kiri/kanan menandai estimasi titik di luar
panel. Segitiga ke atas menandai dukungan n < 20, bukan pemotongan CI.
Seluruh model tetap ditampilkan; pembesaran tidak memilih hasil signifikan.

## Caption yang harus ikut masuk naskah

Titik menunjukkan selisih rata-rata dan garis menunjukkan CI bootstrap
95% marginal untuk selisih rata-rata. Bintang berasal dari uji Wilcoxon
signed-rank dua sisi setelah koreksi Holm, bukan pengujian selisih
rata-rata yang sama dengan CI. Karena itu, CI yang melintasi nol tidak
otomatis bertentangan dengan bintang. k/n = pasangan berubah/tersedia.
I1–I3 adalah variasi isi warmup, bukan tingkat implisit yang berurutan.

Caption per gambar tersedia dalam README serta kolom caption pada indeks:

- [Indeks gabungan GSM8K](../../../result/analyses/linguistic_markers/runs/gender_gsm8k_bab6_v3_1_20261007_234710_219408/combined_figure_index.csv).
- [Indeks gabungan ETHICS](../../../result/analyses/linguistic_markers/runs/gender_ethics_bab6_v3_1_20261007_234752_514764/combined_figure_index.csv).

Indeks gabungan menunjuk karakteristik v3.1 dan heatmap/coverage v3 yang
tetap utuh. Gunakan path dalam indeks, bukan mengambil semua gambar dari
satu folder lama. Tabel `all_comparisons_native_units.csv` di setiap run
menjadi rujukan angka; statistiknya sama dengan sumber v2/v3.

## Pemeriksaan satu halaman cetak

- [ETHICS: PDF A4 satu halaman dengan grafik padat dan caption](../../../result/analyses/linguistic_markers/runs/gender_ethics_bab6_v3_1_20261007_234752_514764/pratinjau_A4_caption_zoom.pdf).
- [ETHICS: PNG pratinjau halaman](../../../result/analyses/linguistic_markers/runs/gender_ethics_bab6_v3_1_20261007_234752_514764/pratinjau_A4_caption_01.png).
- [GSM8K: PDF A4 dengan caption](../../../result/analyses/linguistic_markers/runs/gender_gsm8k_bab6_v3_1_20261007_234710_219408/pratinjau_A4_caption_zoom.pdf).

Cetak pada **100% / actual size**, bukan fit-to-page. Lebar gambar 14 cm;
bidang teks A4 mengikuti margin dalam 4 cm, luar 3 cm, atas/bawah 3 cm.
Pemeriksaan layar tidak menggantikan pemeriksaan cetak fisik.
Naskah LaTeX belum diubah.

## Gambar pronomina lama tetap tersedia

- [Prevalensi pronomina gender menurut model/persona](../img/12_gender_pronoun_prevalence_by_model_persona.png).
- [Pronomina gender: Man vs Woman](../img/13_gender_pronoun_direct_man_vs_woman_forest.png).
- [Paket kandidat lama: profil pronomina](candidate/figures/gsm8k_07_pronoun_profile.png).
- [Paket kandidat lama: perbandingan langsung](candidate/figures/gsm8k_08_pronoun_direct.png).

Keempat gambar dipertahankan tanpa perubahan dan dicatat hash-nya dalam
`preserved_pronoun_figures.csv`. Gambar tersebut **historis**, bukan hasil
pengukuran ulang v3.1. Notebook pendukung pronomina masih merujuk struktur
folder hasil lama; jalur tersebut belum diperbaiki. Jangan menyamakan
angka historisnya dengan tabel v3.1 tanpa rekonsiliasi sumber.

Rincian pemeriksaan: [laporan v3.1](../../../docs/gender_lm_zoom_caption_20261007.md).
Untuk heatmap yang tidak berubah: [panduan v3](MULAI_DI_SINI_BAB6_V3_20261007.md).
