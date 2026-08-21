# Runbook Selective Rerun GSM8K Regional Cultural

**Tanggal:** 14 Agustus 2026  
**Konfigurasi:** `temperature=0`, `max_tokens=4096`  
**Scope terkini:** seluruh selective rerun selesai; OLMo tidak termasuk

## 1. Target yang benar-benar perlu rerun

Folder `result/rerun_manifests/gsm8k_regional_cultural/pending/` sekarang
kosong. Status final Qwen:

| Model | Target | Masalah |
|---|---:|---|
| Qwen3 Next 80B | 57 uncommitted | 54 truncation dan 3 provider error; coverage 99,7299%, 16/16 persona ≥99% |

Folder `closed_audit/` berisi seluruh enam model. CSV di sana hanya bukti audit.
Llama 3.1 selesai pada coverage
99,9194%. Llama 3.3 selesai dari sisi teknis pada coverage 99,2039%; sisa 168
baris merupakan genuine model non-answer, bukan kegagalan API/parser.

Masalah pada manifest pending adalah kegagalan generasi, bukan jawaban yang
sekadar salah secara matematika. Provider error berarti API mengembalikan
error/tidak menghasilkan final content. `finish_reason=length` berarti model
mencapai batas token sebelum respons selesai. `content_truncated` berarti isi
tampak terputus meskipun metadata tidak selalu menulis `length`.

Llama 3.3 juga mempunyai 146 respons non-answer di luar manifest. Respons ini
berisi persona chatter, penolakan, atau tidak memiliki jawaban matematika
final. Respons tersebut tidak di-rerun dan tidak dipaksa parse.

## 2. Sifat aman runner

- hanya membuka path yang tercantum dalam manifest;
- menolak model, scope, atau path yang tidak cocok;
- membuat backup sebelum setiap request;
- hanya mengganti JSON mentah bila API memberikan final content sukses;
- attempt yang error/kosong disimpan terpisah dan tidak menimpa sumber;
- dapat di-resume; target yang sudah sukses otomatis dilewati;
- tidak membawa ground truth ke request.

Credential OpenRouter untuk ketiga model sudah terdeteksi di
`config/model_keys.json`. Alternatifnya, set `OPENROUTER_API_KEY` di environment
terminal tanpa menuliskan nilainya ke command history proyek.

## 3. Arsip perintah Llama 3.1 8B (sudah selesai)

Jalankan smoke satu target:

```powershell
Set-Location D:\TugasAkhir
python scripts\rerun_llama31_8b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.1-8b-instruct.csv `
  --expected-count 134 --temperature 0 --max-tokens 4096 `
  --execute --limit 1 --concurrency 1
```

Jika smoke berstatus `SAVED`, lanjutkan seluruh target. Target smoke akan
otomatis dilewati:

```powershell
python scripts\rerun_llama31_8b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.1-8b-instruct.csv `
  --expected-count 134 --temperature 0 --max-tokens 4096 `
  --execute --concurrency 4
```

Refresh CSV kanonik:

```powershell
python -m src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona.run `
  --model meta-llama_llama-3.1-8b-instruct `
  --reparse_results result\GMS8K\reports\Regional_Cultural_Persona\meta-llama_llama-3.1-8b-instruct\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.1-8b-instruct.csv
```

## 4. Arsip perintah Llama 3.3 70B (technical rerun selesai)

```powershell
# Smoke satu target.
python scripts\rerun_llama33_70b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.3-70b-instruct.csv `
  --expected-count 301 --temperature 0 --max-tokens 4096 `
  --execute --limit 1 --concurrency 1

# Full/resume setelah smoke SAVED.
python scripts\rerun_llama33_70b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.3-70b-instruct.csv `
  --expected-count 301 --temperature 0 --max-tokens 4096 `
  --execute --concurrency 4

# Refresh CSV kanonik.
python -m src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona.run `
  --model meta-llama_llama-3.3-70b-instruct `
  --reparse_results result\GMS8K\reports\Regional_Cultural_Persona\meta-llama_llama-3.3-70b-instruct\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\meta-llama_llama-3.3-70b-instruct.csv
```

## 5. Perintah Qwen3 Next 80B

```powershell
# Smoke satu target.
python scripts\rerun_qwen3_next_80b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\history\qwen_reasoning1024_round_108.csv `
  --expected-count 108 --temperature 0 --max-tokens 4096 `
  --execute --limit 1 --concurrency 1

# Full/resume setelah smoke SAVED.
python scripts\rerun_qwen3_next_80b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\history\qwen_reasoning1024_round_108.csv `
  --expected-count 108 --temperature 0 --max-tokens 4096 `
  --execute --concurrency 4

# Refresh CSV kanonik.
python -m src.MAIN_PARSING.GSM8K.Regional_Cultural_Persona.run `
  --model qwen_qwen3-next-80b-a3b-thinking `
  --reparse_results result\GMS8K\reports\Regional_Cultural_Persona\qwen_qwen3-next-80b-a3b-thinking\results.csv `
  --refresh_manifest result\rerun_manifests\gsm8k_regional_cultural\history\qwen_initial_1039.csv
```

Jika run Qwen masih menghasilkan `finish_reason=length`, runner sebenarnya
sudah mencoba lima kali (`--max-retries 4`). Karena Qwen thinking dapat
menghabiskan mayoritas 4.096 token untuk reasoning, resume hanya target gagal
dengan reasoning dibatasi 1.024 token:

```powershell
python scripts\rerun_qwen3_next_80b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\history\qwen_reasoning1024_round_108.csv `
  --expected-count 108 --temperature 0 --max-tokens 4096 `
  --reasoning-max-tokens 1024 --execute --concurrency 4
```

Target yang sudah `SAVED` otomatis dilewati, sehingga command ini hanya
mengulang target yang masih gagal. Jangan memakai `--force`.

Round tersebut menyelesaikan 45/108 target. Pada 63 sisanya, provider memakai
rata-rata 3.458 reasoning token walaupun diminta maksimal 1.024. Karena hard
cap diabaikan, uji satu target residual dengan effort rendah sebelum full:

```powershell
python -u scripts\rerun_qwen3_next_80b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\qwen_unresolved_after_retries_63.csv `
  --expected-count 63 --temperature 0 --max-tokens 4096 `
  --reasoning-effort low --execute --limit 1 --concurrency 1 --max-retries 0
```

Hanya bila smoke berstatus `SAVED`, jalankan full/resume:

```powershell
python -u scripts\rerun_qwen3_next_80b_openrouter.py `
  --scope regional --dataset gsm8k `
  --manifest result\rerun_manifests\gsm8k_regional_cultural\closed_audit\qwen_unresolved_after_retries_63.csv `
  --expected-count 63 --temperature 0 --max-tokens 4096 `
  --reasoning-effort low --execute --concurrency 4
```

Jalankan ketiga model secara berurutan, bukan bersamaan, agar progress dan
rate-limit mudah diawasi. Jika proses berhenti, ulangi perintah full model yang
sama; runner akan melanjutkan target yang belum sukses.

## 6. Setelah ketiga model selesai

Bangun ulang agregasi enam model:

```powershell
python -m src.MAIN_PARSING.GSM8K._core.combine `
  --group Regional_Cultural_Persona
```

Angka baru berada pada `accuracy.csv` dan `coverage.csv` di folder laporan
masing-masing model. Jangan memakai jumlah `SAVED` saja sebagai kesimpulan;
status akhir harus dibaca dari CSV setelah refresh parser.
