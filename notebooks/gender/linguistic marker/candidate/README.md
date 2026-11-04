# Candidate Gender GSM8K dan ETHICS

[Buka galeri](index.html). Dua notebook terpisah, satu folder figures utama.

- [GSM8K](GSM8K_Gender_Linguistic_Markers_candidate.ipynb): 8 gambar utama + 1 detail.
- [ETHICS](ETHICS_Gender_Linguistic_Markers_candidate.ipynb): 7 gambar utama + 9 detail.

## Keputusan visual dan batas metodologi

- Cerita utama dipisahkan dari lampiran lengkap, bukan menghapus marker.
- Model memakai warna tetap dan posisi titik terpisah. Tidak ada berlian median lintas-model.
- Lebar cetak 16 cm; DejaVu Sans minimum 9 pt; PNG 240 dpi dan SVG tersedia.
- GSM8K: semua 16 indikator numerik utama ditampilkan; dukungan pasangan tercatat per titik. Titik kosong menandai <20 pasangan berubah, bukan ketidaksignifikanan. Rank-biserial tidak diberi CI median-delta.
- ETHICS: 11 indikator ringkasan dipilih di muka, lalu semua 41 marker memiliki gambar detail. Prevalence profil tetap persen respons; efek density per100kata dan prevalence pp dipisah. Zoom B08 hanya I1–I3 dan dinyatakan jelas; sumber nilainya identik.
- CI ETHICS 95% marginal, bukan CI terkoreksi Holm; keputusan inferensial tetap adjusted_p sumber.
- Profil mencakup semua penjelasan tersedia, termasuk abstain. Missing/technical-only tidak menjadi nol. ETHICS verdict-only tidak masuk denominator marker, tetapi 0 kata pada tabel panjang respons sumber.
- Raw count dan denominator disimpan pada tabel profil/marker, bukan hanya density. Sumber kamus dan adaptasi sendiri tetap dicantumkan di tabel katalog.
- Overall ETHICS bukan akurasi gabungan. Komposisi subset/pasangan dan bobot sumber dipertahankan; tabel per subset tetap ada di sumber.
- Token dan contoh kalimat adalah ilustrasi eksploratif/purposif. Referen gender belum selesai diaudit. Kedua instrumen masih candidate; visualisasi bukan bukti validitas.

## Sumber beku

- [gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817](../../../../result/analyses/linguistic_markers/runs/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817)
- [gender_gsm8k_full_source_aligned_v1_2_1_20260817](../../../../result/analyses/linguistic_markers/runs/gender_gsm8k_full_source_aligned_v1_2_1_20260817)
- [gender_lm_v2_candidate_20260916_r3](../../../../result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260916_r3)
- [gender_lm_presentation_v2_bundle_20260920_r2](../../../../result/benchmarks/ethics/reports/gender_lm_presentation_v2_bundle_20260920_r2)

## Reproduksi

Dari `D:\TugasAkhir`, jalankan dengan **path output baru**:

```powershell
python -m src.MAIN_PARSING.ETHICS.build_gender_lm_candidates --output "notebooks/gender/linguistic marker/candidate_rerun"
```

Builder menolak folder yang sudah ada. Notebook membaca tabel dan menanamkan PNG; builder di atas merender ulang semuanya dari sumber beku dan mengeksekusi kedua notebook. Tidak ada API/model call atau perubahan aturan ekstraksi.

`source_hashes.json` membandingkan sumber dan artefak lama sebelum/sesudah; `code_hashes.json` mengunci kode pembuat; `verification.json` mencatat jumlah gambar dan hasil eksekusi.
