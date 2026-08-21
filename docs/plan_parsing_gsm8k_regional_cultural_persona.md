# Parsing GSM8K — Regional Cultural Persona

**Tanggal status:** 14 Agustus 2026  
**Input mentah:** `result/GMS8K/regional_cultural_persona/`  
**Laporan kanonik:** `result/GMS8K/reports/Regional_Cultural_Persona/`  
**Manifest rerun:** `result/rerun_manifests/gsm8k_regional_cultural/`  
**Perintah rerun:** `docs/run_gsm8k_regional_cultural_rerun.md`

Dokumen ini adalah sumber status parsing Regional Cultural. Dokumen
`plan_gsm8k_regional_cultural_persona.md` tetap menjadi catatan rencana
generasi lama, bukan status hasil parsing terkini.

## 1. Scope dan keputusan metodologis

- Scope utama terdiri dari **6 model × 16 persona × 1.319 soal = 126.624
  baris**.
- Model yang dipakai sama dengan Religion Persona: DeepSeek R1 Distill Qwen
  32B, Llama 3.1 8B, Llama 3.3 70B, Ministral 8B, GPT-OSS 20B, dan Qwen3 Next
  80B.
- **OLMo tidak masuk scope.** Data OLMo yang parsial adalah eksperimen lama;
  data itu tidak diparse, tidak diagregasi, dan tidak dibuatkan manifest
  rerun.
- Ada empat region, masing-masing satu persona eksplisit dan tiga persona
  implisit: North American, East Asian, South Asian, dan Sub-Saharan African.
- Parser hanya menilai `response.content`. Reasoning terpisah tidak dipakai.
- Ekstraktor tidak menerima atau membaca ground truth.
- Tidak ada rescue berdasarkan question ID, nama persona, atau region.
- Rule pack boleh berbeda per model, tetapi tidak boleh berbeda per persona.
- Akurasi memakai seluruh 21.104 unit per model sebagai penyebut. Coverage
  selalu dilaporkan berdampingan dengan akurasi.
- Selective rerun memakai `temperature=0` dan `max_tokens=4096`.
- Batas selesai mengikuti Religion Persona: semua persona memiliki coverage
  minimal 99%; technical failure yang tersisa boleh diterima bila di bawah
  0,5%, tetap dicatat sebagai missing dan tidak diimputasi.

## 2. Hasil baseline kanonik

| Model | Akurasi | Coverage | Akurasi core | Coverage core | Persona ≥99% | Status |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek R1 Distill Qwen 32B | 93,8400% | 99,6067% | 67,1958% | 71,1666% | 16/16 | selesai |
| Llama 3.1 8B | 82,0555% | 99,9194% | 25,4265% | 30,8520% | 16/16 | selesai |
| Llama 3.3 70B | 92,7644% | 99,2039% | 56,8139% | 60,1829% | 11/16 | technical rerun selesai; 168 model non-answer |
| Ministral 8B | 90,9259% | 99,9526% | 78,9992% | 85,4435% | 16/16 | selesai |
| GPT-OSS 20B | 91,8641% | 99,6162% | 65,5089% | 70,3326% | 16/16 | selesai |
| Qwen3 Next 80B | 95,5933% | 99,7299% | 91,1581% | 94,4560% | 16/16 | selesai |

DeepSeek, Llama 3.1, Ministral, dan GPT-OSS memenuhi coverage seluruh persona
minimal 99%. Technical rerun Llama 3.3 juga sudah tuntas dengan nol kegagalan
API, tetapi lima persona tetap di bawah 99% karena genuine model non-answer;
baris tersebut dipertahankan dan tidak diimputasi. Target operasional yang
masih perlu rerun sekarang tidak ada; Qwen juga sudah selesai pada coverage
99,7299% dengan 16/16 persona di atas 99%.

## 3. Penyebab baris belum ter-cover dan manifest rerun

Manifest hanya berisi kegagalan generasi yang layak diulang, bukan semua
abstention parser. Target operasional sudah dipisah secara fisik:

- `pending/`: kosong — tidak ada selective rerun wajib yang tersisa;
- `closed_audit/`: seluruh enam model, termasuk residual Qwen — simpan sebagai
  audit dan jangan dijalankan lagi.

| Model | Target | Rincian |
|---|---:|---|
| DeepSeek | 64 | 64 `finish_reason_length` |
| Llama 3.1 | 134 | 134 `finish_reason_length` |
| Llama 3.3 | 301 | 201 `provider_error`, 78 `finish_reason_length`, 22 `content_truncated` |
| Ministral | 7 | 7 `finish_reason_length` |
| GPT-OSS | 67 | 47 `finish_reason_length`, 20 `provider_error` |
| Qwen | 1.039 | 549 `provider_error`, 474 `finish_reason_length`, 16 `finish_reason_error` |

Pada Llama 3.3 masih ada 146 respons non-generation yang tidak di-commit.
Audit isi menunjukkan respons tersebut umumnya berisi persona chatter,
penolakan, atau tidak memberi jawaban matematika final. Itu bukan parser/API
failure dan tidak dipaksa menjadi angka. Pada Qwen, seluruh 1.039 baris yang
belum di-commit merupakan kegagalan generasi dan sudah tercakup manifest.

Setiap manifest mengunci `rerun_temperature=0.0` dan `max_tokens=4096`, tidak
memuat ground truth, dan menunjuk tepat ke JSON sumber yang boleh diganti.
Manifest `pending/` juga sudah lulus dry-run pada runner API masing-masing.

## 4. Kode parser

Parser Regional Cultural berada di:

- `src/MAIN_PARSING/GSM8K/Regional_Cultural_Persona/personas.py` — daftar 16
  persona dan enam model; OLMo dikecualikan secara eksplisit;
- `src/MAIN_PARSING/GSM8K/Regional_Cultural_Persona/run.py` — CLI per model;
- `src/MAIN_PARSING/GSM8K/Regional_Cultural_Persona/rules_*.py` — satu rule
  pack terpisah untuk setiap model;
- `src/MAIN_PARSING/GSM8K/_core/` — ekstraksi, normalisasi, scoring, agregasi,
  dan validasi bersama;
- `scripts/run_gsm8k_regional_parsing_all.py` — menjalankan enam parser;
- `scripts/build_gsm8k_religion_coverage_manifest.py` — builder generik
  Religion/Regional melalui `--group` dan `--raw-group`.

Rule pack Regional mewarisi aturan model Religion sebagai baseline, lalu hanya
menambahkan perilaku yang sudah diaudit pada data Regional. Fallback terminal
bold aktif khusus Ministral dan GPT-OSS:

- Ministral: 7 commit baru, 7/7 cocok dengan reference, dan tidak mengubah
  commit lama;
- GPT-OSS: 34 commit baru, 31 cocok dengan reference; tiga sisanya adalah
  jawaban eksplisit model `106.12` ketika reference `106`, sehingga benar
  dicatat sebagai kesalahan model, bukan kesalahan parsing.

Fallback yang lebih luas tidak dipakai karena pengujian menunjukkan regresi
correct-to-wrong pada model lain.

## 5. Perintah parsing dan agregasi

Jalankan semua model. Default sengaja satu model pada satu waktu agar tidak
berebut disk; pembacaan 64 JSON dilakukan paralel di dalam model.

```powershell
python scripts\run_gsm8k_regional_parsing_all.py
```

Jalankan satu model:

```powershell
python -m src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona.run `
  --model meta-llama_llama-3.1-8b-instruct
```

Jalankan baseline core yang identik untuk analisis sensitivitas:

```powershell
python scripts\run_gsm8k_regional_parsing_all.py --rules core
```

Gabungkan enam model:

```powershell
python -m src.MAIN_PARSING.GSM8K._core.combine `
  --group Regional_Cultural_Persona
```

Setelah selective rerun mengganti JSON target suatu model, refresh hanya baris
manifest lalu hitung ulang CSV. Contoh Llama 3.1:

```powershell
python -m src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona.run `
  --model meta-llama_llama-3.1-8b-instruct `
  --reparse_results result\GMS8K\reports\Regional_Cultural_Persona\meta-llama_llama-3.1-8b-instruct\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_regional_cultural\pending\meta-llama_llama-3.1-8b-instruct.csv
```

Perintah smoke, full/resume, dan refresh untuk ketiga model tersedia lengkap
di `docs/run_gsm8k_regional_cultural_rerun.md`. Jangan rerun seluruh 21.104
unit jika manifest selektif tersedia.

## 6. Artefak hasil

Setiap folder model kanonik berisi tepat 22 berkas:

- `results.csv`, `accuracy.csv`, dan `coverage.csv`;
- `results_core.csv`, `accuracy_core.csv`, dan `coverage_core.csv`;
- 16 berkas `results_<persona>.csv`.

Folder `_combined` berisi:

- `all_models_results.csv` — 126.624 baris;
- `all_models_accuracy.csv` dan `all_models_accuracy_core.csv`;
- `all_models_coverage.csv`;
- `ranking_sensitivity.csv`.

Peringkat rule pack model adalah DeepSeek, GPT-OSS, Llama 3.3, Qwen,
Ministral, lalu Llama 3.1. Peringkat core adalah Qwen, Ministral, DeepSeek,
GPT-OSS, Llama 3.3, lalu Llama 3.1. Karena peringkat berubah, sensitivitas
parser ini wajib dilaporkan dalam naskah TA.

## 7. Validasi yang sudah lulus

- enam model masing-masing tepat 21.104 baris;
- setiap model tepat 16 persona × 1.319 baris;
- pasangan `(persona, question_id)` unik;
- 16 CSV split identik dengan subset `results.csv`;
- scoring dan ringkasan CSV cocok ketika dihitung ulang;
- V2 no-gold-peek: 1.600 sampel per model tetap identik ketika ground truth
  dikosongkan;
- V6: tidak ada rescue literal question ID;
- V7: tidak ada cabang berdasarkan nama region/persona;
- output pembacaan JSON serial dan paralel identik byte-for-byte.

## 8. Urutan pekerjaan berikutnya

1. Llama 3.1 selesai: 134/134 target berhasil di-commit dan 16/16 persona
   mencapai 99%.
2. Llama 3.3 technical rerun selesai: 279/301 target berhasil di-commit; 22
   respons baru dan 146 respons lama tetap genuine non-answer. Jangan dipaksa
   parse atau di-rerun lagi.
3. Qwen selesai: coverage 99,7299%, seluruh persona di atas 99%, dengan 57
   uncommitted tersisa (54 truncation dan 3 provider error). Tidak diimputasi.
4. Hentikan per model segera setelah seluruh persona mencapai 99% dan sisa
   technical failure kurang dari 0,5%.
5. Bangun ulang `_combined` dan catat angka final di naskah.
