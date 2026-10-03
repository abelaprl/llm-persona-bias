# Combined GSM8K Gender dan Religion yang terkunci sumber

Tanggal build: 17 Agustus 2026

## Keputusan sumber

Combined baru tidak memakai `result/archive/reports_mixed_legacy/all_results_combined.csv` karena
file tersebut tidak mempunyai teks respons lengkap untuk lima dari enam model
utama dan key-nya tidak stabil. Sumber dipilih mengikuti provenance analisis
publikasi v5:

- Gender: 54 assignment report tingkat pertanyaan di
  `result/archive/reports_mixed_legacy/<model>/`; GPT-OSS Neutral berasal dari aggregate model karena
  `results_neutral.csv` tidak tersedia.
- Religion: enam report kanonis
  `result/benchmarks/gsm8k/reports/Religion_Persona/<model>/results.csv`.
- Path dan SHA-256 setiap sumber harus sama dengan
  `persona_publication_analysis_v5_paired_significance_six_models/tables/source_manifest.csv`.

Tidak ada jawaban yang direparse dan tidak ada API call. `question_id` stabil
diturunkan dari filename `gsm8k_XXXXX.json`.

## Output

- Gender: 71.215 baris tersedia di
  `result/benchmarks/gsm8k/reports/Gender_Persona/publication_source_aligned_v1_20260817/_combined/`.
- Religion: 126.624 baris (termasuk baris missing-content eksplisit) tersedia di
  `result/benchmarks/gsm8k/reports/Religion_Persona/publication_source_aligned_v1_20260817/_combined/`.

Setiap folder memuat:

- `all_models_results_full.csv`: respons penuh dan provenance setiap baris;
- `accuracy_by_persona_v5.csv`: akurasi per persona yang digunakan analisis v5;
- `accuracy_overall_v5.csv`: inventory overall, termasuk akurasi designated dan
  akurasi hasil hitung `correct / evaluated`;
- `source_manifest.csv`: hash sumber yang telah dicocokkan;
- `manifest.json`: versi, kebijakan sumber, jumlah baris, dan hash combined.

## Catatan denominator akurasi

Missing tidak diubah menjadi nol. Pada Gender, unit unavailable memang tidak
ada di report tingkat pertanyaan sehingga `evaluated` dapat kurang dari 1.319.
Pada Religion, combined tetap menyimpan semua 1.319 baris per persona, tetapi
denominator inferensial v5 mengecualikan technical failure/missing. Karena itu
Qwen mempunyai 21.104 baris dan 21.098 evaluated; GPT-OSS mempunyai 21.104
baris dan 21.058 evaluated.

Untuk Gender, `accuracy_overall_v5.csv` sengaja menyimpan dua kolom bila sumber
deskriptif dan hitungan benar/total berbeda sedikit:

- `weighted_accuracy_question_level_pct`: angka designated publikasi;
- `accuracy_from_counts_pct`: audit hitung ulang dari outcome pertanyaan.

Perbedaan tersebut tidak disembunyikan atau dipilih berdasarkan nilai yang
lebih tinggi. Pipeline linguistic marker tidak memfilter respons berdasarkan
`is_correct`, sehingga perbedaan akurasi tidak mengubah corpus teks yang
dianalisis.

## Eksperimen yang tidak dipilih

Pemakaian langsung rule pack Religion pada Gender ditolak sebagai sumber
publikasi. Audit menghasilkan perubahan besar pada Ministral (sekitar 93,28%
menjadi 75,43%), yang membuktikan rule pack tersebut belum tervalidasi lintas
grup. Artefaknya dipertahankan sebagai audit terpisah dan tidak menjadi input
default linguistic marker.
