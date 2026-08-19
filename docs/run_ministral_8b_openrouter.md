# Menjalankan rerun Ministral 8B melalui OpenRouter

Runner: `scripts/rerun_ministral_8b_openrouter.py`  
Model: `mistralai/ministral-8b-2512` (order 41)  
Total manifest: **10.461 JSON**

Semua perintah PowerShell di bawah harus dijalankan dari `D:\TugasAkhir` dan
ditulis dalam satu baris. Jangan menambahkan `\` pada akhir perintah PowerShell.

## 1. Gunakan API key baru

Key yang pernah ditempel ke percakapan harus dicabut. Masukkan key OpenRouter
baru tanpa menampilkannya di terminal:

```powershell
$secureKey = Read-Host "OpenRouter API key baru" -AsSecureString
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
```

Runner memprioritaskan `OPENROUTER_API_KEY` dan tidak mencetak nilai key.

## 2. Validasi tanpa API call

Nonregional, 6.336 target:

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope nonregional
```

Regional Cultural, 4.125 target:

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope regional
```

Validasi yang benar menampilkan `Validation complete` dan tidak menulis file.

## 3. Smoke test satu target

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope regional --execute --limit 1 --concurrency 1
```

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope nonregional --execute --limit 1 --concurrency 1
```

Lanjutkan hanya jika target berstatus `SAVED` atau sudah `SKIPPED` karena sukses.

## 4. Jalankan seluruh target

Regional Cultural:

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope regional --execute --concurrency 4
```

Nonregional:

```powershell
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope nonregional --execute --concurrency 4
```

Itulah dua perintah utama yang mencakup seluruh 10.461 target. Pastikan limit
akun OpenRouter mencukupi sebelum menjalankan seluruh manifest.

## Menjalankan per dataset

```powershell
# Nonregional ETHICS: 2.308
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope nonregional --dataset ethics --execute --concurrency 4
```

```powershell
# Nonregional GSM8K: 4.028
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope nonregional --dataset gsm8k --execute --concurrency 4
```

```powershell
# Regional ETHICS: 3.847
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope regional --dataset ethics --execute --concurrency 4
```

```powershell
# Regional GSM8K: 278
python ".\scripts\rerun_ministral_8b_openrouter.py" --scope regional --dataset gsm8k --execute --concurrency 4
```

## Tidak perlu menghapus JSON lama

Jangan hapus file sebelum rerun. Runner membuat backup, memanggil endpoint, dan
baru mengganti file utama jika respons mempunyai final content non-empty.
Percobaan yang gagal disimpan terpisah sehingga file lama tetap aman.

## Arti status `SAVED`

Runner memakai mode collection-first. `SAVED` berarti endpoint berhasil memberi
final content non-empty. Status itu tidak menjamin `parsed_answer` downstream
sudah terisi; hasil dengan `parsing_failed=true` dapat diperiksa atau direparse
pada tahap terpisah setelah koleksi selesai.

## Resume dan kegagalan baru

Jika masih ada `FAILED`, jalankan ulang perintah scope yang sama. Target sukses
otomatis dilewati. HTTP 408/409/429 dan 5XX sudah dicoba ulang maksimal empat
kali per eksekusi dengan exponential backoff.

Jangan memakai `--force` untuk resume normal, karena opsi itu juga mengulang
target manifest yang sudah sukses.

Backup berada di:

- `result/_rerun_backup/ministral_8b_nonregional`;
- `result/_rerun_backup/ministral_8b_regional`.

Percobaan yang gagal disimpan terpisah di:

- `result/_rerun_attempts/ministral_8b_nonregional`;
- `result/_rerun_attempts/ministral_8b_regional`.

Setelah selesai, hapus key dari environment sesi:

```powershell
Remove-Item Env:OPENROUTER_API_KEY
```

