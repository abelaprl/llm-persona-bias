# Rerun Penuh GSM8K Regional DeepSeek 32B di Vast.ai

Dokumen ini khusus untuk mengulang seluruh 21.104 respons yang membentuk:

```text
result/GMS8K/reports/Regional_Cultural_Persona/
  deepseek_deepseek-r1-distill-qwen-32b/results.csv
```

Hasil lama tidak ditimpa. Raw JSON dan `results.csv` baru ditulis ke:

```text
result/GMS8K/reruns/20260815_deepseek_regional_full_vast/raw/
result/GMS8K/reruns/20260815_deepseek_regional_full_vast/reports/
```

Launcher:

```text
scripts/rerun_gsm8k_regional_deepseek_full_vast.py
```

## Kondisi Vast.ai yang sudah diverifikasi

- Akun CLI terautentikasi dan SSH public key `vast-tugasakhir` terdaftar.
- Template private `vLLM DeepSeek R1 Distill Qwen 32B (TA)` tersedia.
- Template memakai checkpoint BF16 resmi
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`.
- Alias endpoint adalah `deepseek/deepseek-r1-distill-qwen-32b`.
- `VLLM_ARGS` memakai `--max-num-seqs 128`, `--max-model-len 8192`,
  `--gpu-memory-utilization 0.92`, prefix caching, dan reasoning parser
  `deepseek_r1`.
- Container disk template adalah 150 GB.

Saldo saat pemeriksaan 15 Agustus 2026 adalah $2,37. Saldo ini sangat tipis
untuk 21.104 respons ditambah waktu download/startup. Tambah credit sebelum
menyewa agar job tidak berhenti menjelang selesai.

## Urutan wajib

Semua perintah dijalankan dari `D:\TugasAkhir`.

### 1. Plan lokal, tanpa API call

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py --mode plan
```

### 2. Rent setelah saldo cukup

Gunakan template private yang sudah ada dan mesin on-demand berikut:

- verified host dan reliability minimal 99%;
- satu GPU dengan VRAM minimal 96 GB;
- container disk minimal 150 GB;
- direct SSH;
- bandwidth download yang memadai;
- harga dan durasi maksimum diperiksa lagi tepat sebelum menekan Rent.

Penawaran Vast bersifat dinamis. Jangan menyalin offer ID lama tanpa mencari
ulang.

### 3. Preflight

Salin target dan port dari tombol SSH instance:

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py `
  --mode preflight `
  --ssh-target "root@IP_INSTANCE" `
  --ssh-port PORT_SSH `
  --ssh-identity-file "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Preflight menunggu endpoint hingga 30 menit karena checkpoint harus diunduh dan
dimuat. Tidak ada generation call pada tahap ini.

### 4. Smoke: maksimum 3 call

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py `
  --mode smoke `
  --ssh-target "root@IP_INSTANCE" `
  --ssh-port PORT_SSH `
  --ssh-identity-file "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Smoke menjalankan dua respons utama dan satu warmup: satu persona explicit dan
satu implicit. Jangan lanjut bila audit akhir smoke tidak bersih.

### 5. Pilot: maksimum 140 call

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py `
  --mode pilot `
  --pilot-samples 8 `
  --ssh-target "root@IP_INSTANCE" `
  --ssh-port PORT_SSH `
  --ssh-identity-file "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Pilot mencakup 8 soal x 16 persona = 128 respons utama dan maksimum 12 warmup.
File smoke yang sudah lengkap dilewati.

### 6. Full/resume: 21.104 respons utama

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py `
  --mode full `
  --confirm-full `
  --ssh-target "root@IP_INSTANCE" `
  --ssh-port PORT_SSH `
  --ssh-identity-file "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Full menjalankan empat proses dengan persona yang disjoint: North American,
East Asian, South Asian, dan Sub-Saharan African. Default 45 worker per shard
ditujukan untuk template `--max-num-seqs 128` yang sudah diuji sebelumnya.

Command yang sama aman digunakan untuk resume. Respons hanya dianggap selesai
bila JSON valid, `response.success=true`, final content tidak kosong, dan
`finish_reason=stop`. Attempt kosong/terpotong dipindahkan ke
`incomplete_attempts/` sebelum dicoba ulang sehingga tidak hilang.

### 7. Verify tanpa Vast

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py --mode verify
```

Harus menunjukkan 1.319/1.319 file lengkap untuk setiap persona.

### 8. Bangun `results.csv` baru

```powershell
.\.venv\Scripts\python.exe scripts\rerun_gsm8k_regional_deepseek_full_vast.py --mode report
```

Output utama:

```text
result/GMS8K/reruns/20260815_deepseek_regional_full_vast/reports/
  Regional_Cultural_Persona/
  deepseek_deepseek-r1-distill-qwen-32b/results.csv
```

Mode `report` menolak berjalan bila raw output belum lengkap.

### 9. Hentikan biaya

Setelah `verify` dan `report` berhasil, destroy instance Vast. Stop saja masih
dapat menyisakan biaya storage.
