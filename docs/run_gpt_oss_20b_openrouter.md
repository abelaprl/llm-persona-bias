# Menjalankan rerun GPT-OSS-20B melalui OpenRouter

Runner: `scripts/rerun_gpt_oss_20b_openrouter.py`  
Model: `openai/gpt-oss-20b` (order 31)  
Total manifest rerun: **5.676 JSON**

Semua perintah PowerShell harus dijalankan dari `D:\TugasAkhir` dalam satu
baris. Jangan menambahkan `\` pada akhir perintah PowerShell.

## 1. Gunakan API key baru

Key yang pernah ditempel ke percakapan harus dicabut. Masukkan key OpenRouter
baru tanpa menampilkannya di terminal:

```powershell
$secureKey = Read-Host "OpenRouter API key baru" -AsSecureString
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
```

Runner memprioritaskan `OPENROUTER_API_KEY` dan tidak mencetak nilainya.

## 2. Validasi tanpa API call

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope nonregional
```

Hasil validasi nonregional harus menunjukkan **118 target**.

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope regional
```

Hasil validasi regional harus menunjukkan **5.558 target**.

## 3. Smoke test satu target

Mulai dari nonregional yang jauh lebih kecil:

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope nonregional --execute --limit 1 --concurrency 1
```

Kemudian regional:

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope regional --execute --limit 1 --concurrency 1
```

Lanjutkan hanya jika target berstatus `SAVED` atau `SKIPPED` karena sudah sukses.

## 4. Jalankan seluruh target

Nonregional:

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope nonregional --execute --concurrency 4
```

Regional Cultural:

```powershell
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope regional --execute --concurrency 4
```

Pastikan limit akun OpenRouter mencukupi sebelum menjalankan 5.558 target
regional.

## Menjalankan per dataset

```powershell
# Nonregional ETHICS: 86
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope nonregional --dataset ethics --execute --concurrency 4
```

```powershell
# Nonregional GSM8K: 32
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope nonregional --dataset gsm8k --execute --concurrency 4
```

```powershell
# Regional ETHICS: 4.995
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope regional --dataset ethics --execute --concurrency 4
```

```powershell
# Regional GSM8K: 563
python ".\scripts\rerun_gpt_oss_20b_openrouter.py" --scope regional --dataset gsm8k --execute --concurrency 4
```

## Reasoning-only tidak direrun

Sebanyak **2.545 response reasoning-only** berada pada
`model_review_only.csv`. Runner ini tidak membaca file tersebut. Jangan
menggabungkannya ke manifest API sebelum audit/reparse lokal dilakukan.

## Tidak perlu menghapus JSON lama

Runner membuat backup dan hanya mengganti file utama setelah menerima final
content non-empty. Percobaan gagal disimpan terpisah sehingga file lama aman.

## Resume

Jika masih ada `FAILED`, jalankan kembali perintah scope yang sama. Jangan pakai
`--force`; target sukses otomatis dilewati. HTTP 408/409/429 dan 5XX dicoba
ulang maksimal empat kali per eksekusi dengan exponential backoff.

Backup berada di:

- `result/_rerun_backup/gpt_oss_20b_nonregional`;
- `result/_rerun_backup/gpt_oss_20b_regional`.

Percobaan gagal berada di:

- `result/_rerun_attempts/gpt_oss_20b_nonregional`;
- `result/_rerun_attempts/gpt_oss_20b_regional`.

Setelah selesai, hapus key dari sesi:

```powershell
Remove-Item Env:OPENROUTER_API_KEY
```

