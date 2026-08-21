# Handoff Linguistic Marker Gender GSM8K v3 — Direct Man vs Woman

Tanggal: 17 Agustus 2026  
Status komputasi: **selesai dan executed**  
Status ilmiah: **candidate**, menunggu audit manual precision/recall marker

## Tujuan revisi

Versi v3 dibuat karena tesis mengklaim perbedaan gender. Membandingkan efek
Man-vs-Neutral dan Woman-vs-Neutral secara visual belum merupakan uji langsung
Man-vs-Woman. V3 menambahkan empat planned direct contrasts:

- `man_vs_woman_explicit`;
- `man_vs_woman_implicit_1`;
- `man_vs_woman_implicit_2`; dan
- `man_vs_woman_implicit_3`.

Semua pasangan dibentuk pada `model_id x question_id`. Arah effect size selalu
Man minus Woman. I1, I2, dan I3 tidak diagregasi.

## Artefak final

Source run:

`result/linguistic_markers/gender_gsm8k_full_source_aligned_v1_2_1_20260817/`

Bundle final:

`result/linguistic_markers/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817/`

Notebook executed:

`notebooks/linguistic_marker_gender_gsm8k_direct_gender_v3.ipynb`

Notebook suplemen kata gender:

`notebooks/linguistic_marker_gender_words_pronouns.ipynb`

Folder berikut adalah percobaan build yang berhenti saat plotting dan **bukan
hasil final**:

`result/linguistic_markers/gender_gsm8k_bab4_visual_revision_v3_direct_gender_20260817/`

Folder gagal tidak ditimpa atau dihapus untuk menjaga provenance. Hasil final
selalu memakai path dengan penanda `r2`.

## Statistik direct gender

Tabel utama:

- `direct_gender_paired_numeric.csv`: 456 uji numeric;
- `direct_gender_correctness_sensitivity.csv`: 456 uji both-correct;
- `direct_gender_correctness_comparison.csv`: perbandingan arah utama vs sensitivity;
- `direct_gender_paired_binary.csv`: 408 exact McNemar marker-presence;
- `direct_gender_availability.csv`: 48 audit paired availability/technical-only;
- `direct_gender_opening_multinomial.csv`: 24 Stuart-Maxwell; dan
- `paired_gender_pronoun_direct.csv`: 48 uji pronoun suplemen.

Koreksi Holm dilakukan dalam keluarga empat kondisi untuk setiap model dan
outcome. Numeric outcomes memakai Wilcoxon signed-rank atau sign test sesuai
diagnostik sparsity/symmetry, dengan rank-biserial dan 95% bootstrap CI.
Binary presence dan availability memakai exact McNemar dengan paired mean
delta serta 95% bootstrap CI. Opening diuji bersama sebagai outcome
komposisional dengan Stuart-Maxwell marginal homogeneity.

## Figur baru dan figur yang diperbaiki

- Gambar 08 menyembunyikan kategori opening dengan total hit nol. `refusal`
  GSM8K bernilai nol dan tidak lagi muncul dalam legend.
- Gambar 12 memakai dua cabang N-E-I1-I2-I3: garis Man dan Woman tidak lagi
  disambungkan menjadi satu trayektori.
- Gambar 13 menguji pronoun Man-vs-Woman secara langsung.
- Gambar 14 adalah direct gender numeric volcano.
- Gambar 15 menunjukkan heterogenitas arah pada enam model untuk 16 primary
  outcomes.
- Gambar 16 adalah paired differential-availability forest.
- Gambar 17 menunjukkan direct gender opening-composition effect.

Kalimat defensif seperti “bukan uji hipotesis” tidak diletakkan di dalam
gambar. Gambar hanya menjelaskan perbandingan, arah, satuan, dan encoding.

## Angka audit penting

- Primary numeric direct tests: 384.
- Primary numeric `p_holm < 0,05`: 56; seluruhnya memiliki minimal 20 pasangan
  nonzero.
- Significant binary-presence tests: 46 dari 408.
- Significant availability rows: 9 dari 48.
- Significant opening-composition tests: 9 dari 24.
- Significant direct-pronoun tests: 4 dari 48; statusnya suplemen.
- Median paired retention pada sensitivity both-correct: 89,8%–90,9%.
- Konkordansi arah primary outcome dengan minimal 20 perubahan pada kedua
  analisis: E 83,1%; I1 85,2%; I2 88,3%; I3 96,7%.

Differential availability paling besar terdapat pada GPT-OSS I1: teks tersedia
Man 86,81% dan Woman 92,49%, delta -5,69 pp, 95% CI [-7,51, -3,94],
`p_holm=1,73e-9`. Pola komplemennya terlihat pada technical-only: Man 13,19%
dan Woman 7,51%, delta +5,69 pp. Karena itu efek linguistik GPT-OSS, terutama
I1, harus dibaca bersama selection akibat availability.

`refusal` berjumlah nol pada seluruh opening GSM8K. Kritik yang menghubungkan
warna refusal dengan missingness tidak didukung tabel aktual. Refusal tetap
relevan untuk benchmark Ethics, tetapi bukan outcome informatif pada run GSM8K
ini.

## Batas klaim

- Direct effect bersifat model-specific; tidak ada pooled causal effect enam
  model.
- Signifikansi tidak cukup tanpa effect size dan support `n_nonzero`.
- Pronoun dapat merujuk tokoh dalam soal GSM8K, sehingga tidak boleh dipakai
  sendirian sebagai bukti adopsi identitas persona.
- Word cloud dipindahkan menjadi suplemen dan tidak muncul dalam notebook utama
  v3.
- Audit manual marker masih kosong, sehingga hasil belum boleh dipromosikan
  menjadi registry final.

## Validasi

- 54 unit tests lulus.
- 17 PNG tercatat dan nama figur unik.
- Notebook v3 disusun ulang dengan alur visual v2: 26 cell bergantian antara
  penjelasan singkat dan hasil yang relevan.
- 13 dari 13 code cells notebook v3 executed.
- Notebook v3 memiliki 16 embedded images dan tidak mempunyai error output.
- Bundle final `r2` dipilih secara otomatis; folder build yang tidak lengkap
  tidak digunakan.
- Key numeric, binary, availability, opening, dan pronoun direct semuanya unik.
- Seluruh `p_holm` berada pada [0, 1].
- Seluruh CI numeric/binary/availability memuat point estimate.
- Jumlah pasangan both-correct tidak pernah melebihi analisis utama.
- Word cloud tidak ditampilkan di notebook utama v3.

## Kode

- `src/MAIN_PARSING/LINGUISTIC_MARKERS/statistics.py`
- `src/MAIN_PARSING/LINGUISTIC_MARKERS/tests/test_loaders_and_statistics.py`
- `scripts/build_gender_lm_visual_revision_v2.py`
- `scripts/build_gender_lm_visual_revision_v3.py`
- `scripts/build_gendered_lexicon_analysis.py`
- `scripts/create_gender_lm_visual_notebook_v3.py`
