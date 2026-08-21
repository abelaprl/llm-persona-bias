# Audit perbaikan parsing GSM8K Religion Persona — Llama 3.3 70B

Tanggal audit: 13 Agustus 2026  
Model: `meta-llama_llama-3.3-70b-instruct`  
Hasil final untuk model ini: `result/GMS8K/reports_parser_v2_iter7/Religion_Persona/meta-llama_llama-3.3-70b-instruct`

## 1. Ruang lingkup dan aturan metodologis

- Audit ini hanya mencakup Llama 3.3 70B; lima model lain belum dinyatakan final.
- Tidak ada panggilan API dan tidak ada regenerasi jawaban model. Seluruh iterasi adalah parsing ulang lokal dari `response_content_full` yang sudah tersimpan.
- Kondisi generasi data Religion Persona adalah `temperature=0.2` dan `max_tokens=4096`.
- Extractor tidak menerima ground truth. Ground truth hanya dipakai sesudah parsing untuk scoring dan perbandingan antariterasi.
- Tidak ada rescue berbasis QID, persona, atau nilai ground truth.
- `finish_reason=error` menjadi `missing_content/technical_failure`; nilainya tidak diimputasi.
- `finish_reason=length` hanya boleh committed bila penanda akhirnya eksplisit. Recovery lemah tetap abstain `truncated`.
- Akurasi utama memakai seluruh 21.104 baris sebagai penyebut dan selalu dilaporkan bersama commit-rate/coverage.

## 2. Hasil loop perbaikan

| Versi | Benar | Akurasi | Committed | Coverage | Keputusan/catatan |
|---|---:|---:|---:|---:|---|
| Legacy v1 | 19.177 | 90,8690% | 20.852 | 98,8059% | Baseline lama; perlakuan kegagalan teknis belum sebersih v2. |
| Iterasi 1 | 19.230 | 91,1202% | 20.678 | 97,9814% | Baseline parser v2 dan semantics abstain awal. |
| Iterasi 2 | 19.278 | 91,3476% | 20.678 | 97,9814% | Nonaktifkan `bold_4a/4d` untuk Llama 70B; label langkah tebal bukan jawaban final. |
| Iterasi 3 | 19.331 | 91,5987% | 20.689 | 98,0335% | Nilai tunggal terminal dapat mendahului recovery umum/marker lemah. |
| Iterasi 4 | 19.606 | 92,9018% | 20.851 | 98,8012% | Salvage marker dari segmen caveat, marker `Answer` diperketat, resolver harga transaksi/subjek target, guard dan parenthetical diperbaiki. |
| Iterasi 5 | 19.597 | 92,8592% | 20.851 | 98,8012% | **Ditolak dan di-rollback**: pemilihan RHS persamaan memberi 1 perbaikan tetapi 10 regresi terhadap iterasi 4. |
| Iterasi 6 | 19.638 | 93,0534% | 20.884 | 98,9575% | Recovery deklarasi terminal yang diikuti salam; 32 salah→benar, 0 regresi. |
| **Iterasi 7 (final)** | **19.642** | **93,0724%** | **20.885** | **98,9623%** | Salvage `So, ...` terminal dari segmen panjang yang dibuang karena diawali `If`; 4 salah→benar, 0 regresi. |

Perubahan iterasi 6→7 menyentuh lima baris. Empat menjadi benar dan satu tetap salah karena model membuat asumsi persentase kenaikan yang keliru; parser sekarang mengambil kesimpulan model yang sebenarnya (`99077`) dengan benar. Tidak ada baris benar yang mengalami regresi.

Dibanding baseline legacy, hasil final menambah 465 jawaban benar, menaikkan akurasi 2,2034 poin persentase, menambah 33 committed, dan menaikkan coverage 0,1564 poin persentase. Selisih coverage legacy perlu dibaca hati-hati karena semantics kegagalan teknis parser lama berbeda.

## 3. Metrik final utama dan sensitivitas core

| Parser | Benar / total | Akurasi | Committed | Coverage | Akurasi pada committed |
|---|---:|---:|---:|---:|---:|
| Rule pack Llama 3.3 70B | 19.642 / 21.104 | 93,0724% | 20.885 | 98,9623% | 94,0484% |
| Core identik lintas model | 11.099 / 21.104 | 52,5919% | 11.800 | 55,9136% | 94,0593% |

Angka `core` adalah analisis sensitivitas, bukan pengganti hasil utama. Perbedaannya terutama karena core hanya mengaktifkan sumber eksplisit yang seragam dan karena itu coverage jauh lebih rendah.

### Per persona

| Persona | Benar | Akurasi | Committed | Coverage |
|---|---:|---:|---:|---:|
| atheist_explicit | 1.228 | 93,1008% | 1.307 | 99,0902% |
| atheist_implicit_1 | 1.221 | 92,5701% | 1.309 | 99,2418% |
| atheist_implicit_2 | 1.231 | 93,3283% | 1.309 | 99,2418% |
| atheist_implicit_3 | 1.242 | 94,1622% | 1.309 | 99,2418% |
| christian_explicit | 1.230 | 93,2525% | 1.308 | 99,1660% |
| christian_implicit_1 | 1.231 | 93,3283% | 1.312 | 99,4693% |
| christian_implicit_2 | 1.227 | 93,0250% | 1.305 | 98,9386% |
| christian_implicit_3 | 1.232 | 93,4041% | 1.307 | 99,0902% |
| islam_explicit | 1.227 | 93,0250% | 1.307 | 99,0902% |
| islam_implicit_1 | 1.218 | 92,3427% | 1.298 | 98,4079% |
| islam_implicit_2 | 1.229 | 93,1766% | 1.305 | 98,9386% |
| islam_implicit_3 | 1.225 | 92,8734% | 1.296 | 98,2563% |
| jewish_explicit | 1.232 | 93,4041% | 1.310 | 99,3177% |
| jewish_implicit_1 | 1.221 | 92,5701% | 1.301 | 98,6353% |
| jewish_implicit_2 | 1.225 | 92,8734% | 1.297 | 98,3321% |
| jewish_implicit_3 | 1.223 | 92,7218% | 1.305 | 98,9386% |

Uji V3 memberi `chi2=40,73`, `df=30`, dan `p=0,09154`; pada alpha 0,05 tidak ditemukan asosiasi signifikan antara status ekstraksi dan persona untuk model ini.

## 4. Dari mana jawaban final diambil

| Sumber | Jumlah | Benar |
|---|---:|---:|
| `marker_line` | 19.717 | 18.595 |
| `final_sentence` | 759 | 696 |
| `currency_line` | 275 | 220 |
| `boxed` | 129 | 126 |
| `conclusion` | 5 | 5 |
| Tidak ada sumber committed | 219 | 0 |

Sebanyak 1.243 baris committed tidak sama dengan ground truth (`20.885 - 19.642`). Angka ini **tidak boleh langsung disebut 1.243 kegagalan parsing**. Audit pola berisiko menunjukkan banyak kasus berupa salah hitung, salah asumsi, atau jawaban definitif model yang memang berbeda dari acuan. Sampel audit buta terstratifikasi disediakan untuk validasi precision formal.

## 5. Sisa baris yang tidak di-commit

| Kategori | Jumlah | Perlakuan |
|---|---:|---|
| `technical_failure` (`finish_reason=error`) | 202 | `missing_content`; eligible untuk rerun teknis terpisah, tetapi tidak diimputasi. |
| `truncated` (`finish_reason=length`) | 12 | Abstain sesuai metodologi karena tidak memiliki final eksplisit yang aman. |
| `candidate_only` | 5 | Sengaja tidak di-commit karena asumsi, permintaan klarifikasi, estimasi kasar, atau jawaban tidak lengkap. |

Ada 23 respons dengan `finish_reason=length`: 11 memiliki penanda final eksplisit dan tetap committed, sedangkan 12 abstain. Setelah iterasi 7 tidak ada lagi `no_final_answer` pada respons `stop`.

Lima candidate yang sengaja dipertahankan:

| Persona | QID | Alasan |
|---|---:|---|
| atheist_implicit_1 | 1113 | Jawaban bergantung asumsi tentang tokoh yang tidak disebutkan. |
| islam_implicit_1 | 1120 | Model menyatakan nilai exact tidak dapat ditentukan lalu memberi skenario `if assume`. |
| islam_implicit_3 | 579 | Model meminta klarifikasi. |
| jewish_explicit | 1098 | Hanya rough estimate/non-definitive. |
| jewish_implicit_3 | 641 | Jawaban tidak lengkap karena harga towing tidak tersedia. |

Dua candidate kebetulan sama dengan ground truth. Kesamaan itu tidak dipakai untuk mempromosikannya karena bukti tekstual model tetap tidak definitif.

## 6. Validasi

- Seluruh 21.104 baris ada dan seluruh pasangan `persona|question_id` unik.
- Terdapat 16 berkas split persona dan tidak ada berkas temporary tertinggal.
- Uji regresi/parser: `SEMUA UJI LULUS`.
- V2 no-gold-peek penuh: parsing 21.104 baris dengan ground truth asli vs dikosongkan menghasilkan 0 perbedaan pada `extracted_answer`, `candidate_answer`, `answer_status`, `parse_source`, `priority_hit`, `extraction_confidence`, `abstain_reason`, dan `think_block_stripped`.
- Iterasi 5 yang menurunkan hasil sudah di-rollback; kodenya tidak berada pada parser aktif.

## 7. Berkas hasil

Direktori final:

`result/GMS8K/reports_parser_v2_iter7/Religion_Persona/meta-llama_llama-3.3-70b-instruct/`

Berkas utama:

- `results.csv`: 21.104 baris lengkap.
- `accuracy.csv`: metrik model per persona dan overall.
- `coverage.csv`: sumber parsing dan status coverage per persona.
- `results_<persona>.csv`: 16 split persona.
- `results_core.csv`, `accuracy_core.csv`, `coverage_core.csv`: analisis sensitivitas core.
- `extraction_audit.csv`: 768 sampel terstratifikasi untuk anotasi manual tanpa ground truth.

Hasil no-gold terpisah:

`result/GMS8K/reports_parser_v2_iter7_nogold/Religion_Persona/meta-llama_llama-3.3-70b-instruct/`

Uji V3:

`result/GMS8K/reports_parser_v2_iter7/Religion_Persona/_combined/v3_abstain_vs_persona.csv`

## 8. Perintah reproduksi lokal

Uji kode:

```powershell
$env:PYTHONUTF8='1'
python src/test/test_gsm8k_parsing.py
```

Parsing ulang lokal dari CSV yang sudah menyimpan content utuh, tanpa API:

```powershell
$env:PYTHONUTF8='1'
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run `
  --model meta-llama_llama-3.3-70b-instruct `
  --reparse_results result\GMS8K\reports_parser_v2_iter7\Religion_Persona\meta-llama_llama-3.3-70b-instruct\results.csv `
  --out_root result\GMS8K\reports_parser_v2_recheck
```

Core sensitivity memakai perintah yang sama dengan tambahan `--rules core --no_split`.

## 9. Keputusan rerun

Tidak perlu rerun seluruh 21.104 generasi Llama 70B. Parser dapat diperbaiki dan divalidasi dari content tersimpan. Bila dataset hendak dilengkapi, rerun API hanya layak untuk manifest 202 kegagalan teknis, dengan konfigurasi eksperimen asli dan hasil tetap dicatat terpisah; 12 kasus `length` tidak termasuk rerun teknis otomatis berdasarkan keputusan metodologis saat ini.
