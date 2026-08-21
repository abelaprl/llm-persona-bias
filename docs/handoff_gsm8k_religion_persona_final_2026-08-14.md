# Handoff Final Parsing GSM8K — Religion Persona

**Tanggal status:** 14 Agustus 2026  
**Input:** `result/GMS8K/Religion_Persona/`  
**Laporan kanonik:** `result/GMS8K/reports/Religion_Persona/`  
**Jumlah unit:** 6 model × 16 persona × 1.319 soal = 126.624 baris

## 1. Keputusan metodologis yang dipakai

- Teks yang dinilai hanya `response.content`; reasoning terpisah tidak dipakai.
- Ekstraktor tidak menerima atau membaca ground truth.
- Tidak ada forced rescue berdasarkan question ID atau persona.
- Rule pack boleh berbeda per model, tetapi tidak boleh berbeda per persona.
- Akurasi memakai seluruh 21.104 unit per model sebagai penyebut.
- Coverage/commit-rate selalu dilaporkan bersama akurasi.
- Selective rerun memakai `temperature=0` dan `max_tokens=4096`.
- Model dinyatakan selesai ketika seluruh 16 persona memiliki coverage ≥99%.
- Sisa technical failure boleh diterima hanya bila <0,5%; baris tetap missing
  dan tidak diimputasi.

## 2. Hasil kanonik

| Model | Akurasi | Coverage | Technical failure | Persona ≥99% | Status |
|---|---:|---:|---:|---:|---|
| Llama 3.3 70B | 93,9727% | 99,9194% | 0,0000% | 16/16 | selesai |
| Ministral 8B | 87,5569% | 99,7962% | 0,0000% | 16/16 | selesai |
| Llama 3.1 8B | 81,9229% | 99,3508% | 0,0000% | 16/16 | selesai |
| GPT-OSS 20B | 92,1342% | 99,4503% | 0,2180% | 16/16 | selesai |
| Qwen3 Next 80B | 95,5553% | 99,3840% | 0,0284% | 16/16 | selesai |
| DeepSeek R1 Distill Qwen 32B | 90,7316% | 96,9200% | 2,6156% | 2/16 | menunggu rerun |

Lima model sudah memenuhi batas berhenti. DeepSeek belum ditutup karena sisa
kegagalan teknis masih melewati 0,5% dan 14 persona belum mencapai 99%.

## 3. Audit perbaikan DeepSeek

Audit dilakukan pada seluruh 21.104 baris, bukan sampel:

- 39 respons tambahan berhasil di-commit;
- 14 nilai yang sebelumnya salah parsing berubah menjadi benar;
- 0 nilai benar berubah menjadi salah;
- empat commit baru merupakan jawaban eksplisit model yang matematisnya salah,
  sehingga tetap dihitung sebagai kesalahan model;
- dua respons sukses dipulihkan dari objek JSON lengkap yang memiliki sampah
  sesudah penutup objek; berkasnya tetap ditandai `input_json_valid=False`.

Manifest DeepSeek saat ini berisi 647 target unik:

- 551 provider/API error;
- 95 `finish_reason=length`;
- 1 respons yang terpotong berdasarkan isi meskipun metadata tidak menyatakan
  `length`.

Manifest:
`result/rerun_manifests/gsm8k_deepseek/deepseek_religion_coverage_failures.csv`

## 4. Artefak utama

Untuk setiap model, folder kanonik berisi:

- `results.csv` — 21.104 baris;
- `accuracy.csv` dan `coverage.csv`;
- `results_core.csv`, `accuracy_core.csv`, dan `coverage_core.csv`;
- 16 `results_<persona>.csv`.

Kode utama:

- `src/MAIN_PARSING/GSM8K/_core/` — normalisasi, guard, ekstraksi, scoring,
  penulisan CSV, dan validasi bersama;
- `src/MAIN_PARSING/GSM8K/Religion_Persona/rules_*.py` — rule pack per model;
- `src/MAIN_PARSING/GSM8K/Religion_Persona/run.py` — CLI;
- `scripts/build_gsm8k_religion_coverage_manifest.py` — manifest selective
  rerun;
- `scripts/rerun_deepseek_32b_vast.py` — runner DeepSeek melalui vLLM lokal.

## 5. Validasi yang sudah lulus

- struktur 21.104 baris, 16 persona × 1.319, tanpa duplikasi;
- seluruh CSV split identik dengan subset `results.csv`;
- ringkasan `accuracy.csv`/`coverage.csv` dihitung ulang dan cocok;
- V2 no-gold-peek DeepSeek: 3.200/3.200 hasil identik;
- V6: tidak ada literal `gsm8k_XXXXX` di rule pack atau `_core`;
- V7: tidak ada cabang agama/persona di rule pack;
- dry-run runner Qwen dan DeepSeek tidak melakukan API call atau write.

## 6. Melanjutkan DeepSeek

Syarat awal: server vLLM tersedia di
`http://127.0.0.1:8000/v1/chat/completions`.

```powershell
# Validasi manifest saja.
python scripts\rerun_deepseek_32b_vast.py

# Jalankan/resume; temperature dan max_tokens dikunci oleh wrapper.
python scripts\rerun_deepseek_32b_vast.py --execute --concurrency 4

# Refresh target manifest dan hitung ulang laporan kanonik.
python -m src.MAIN_PARSING.GSM8K.Religion_Persona.run `
  --model deepseek_deepseek-r1-distill-qwen-32b `
  --reparse_results result\GMS8K\reports\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_deepseek\deepseek_religion_coverage_failures.csv `
  --out_root result\GMS8K\reports
```

## 7. Aturan untuk persona berikutnya

Regional Cultural memakai enam model yang sama. OLMo tidak masuk scope utama.
Arsitektur `_core` dipakai ulang; setiap model tetap mempunyai rule pack
sendiri dan harus diaudit pada data Regional Cultural sebelum angka ditutup.

