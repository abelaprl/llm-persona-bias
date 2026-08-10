# DeepSeek R1 Distill Qwen 32B di Vast.ai — Runbook Lengkap

Dokumen ini adalah satu-satunya runbook untuk melengkapi eksperimen regional-kultural DeepSeek pada repo ini. Ikuti urutannya dan jangan lompat langsung ke `full`.

Sumber resmi yang menjadi acuan:

- [Vast vLLM template](https://docs.vast.ai/vllm-llm-inference-and-serving)
- [Vast SSH dan port forwarding](https://docs.vast.ai/guides/instances/connect/ssh)
- [Vast Instance Portal](https://docs.vast.ai/guides/instances/connect/instance-portal)
- [vLLM reasoning parser](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)

## 1. SPEC

### Goal

Menjalankan checkpoint BF16 resmi berikut pada Vast.ai:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

Endpoint harus mengiklankan alias yang dipakai seluruh repo:

```text
deepseek/deepseek-r1-distill-qwen-32b
```

### Inputs

- Model order `43` dari `config/model_keys.json`.
- 16 persona dari `inputs/regional_cultural_persona.json`.
- ETHICS: `data/ETHICS_test_4subsets_10percent_from200.json`.
- GSM8K: `data/gsm8k_test.json`.

### Outputs

```text
result/ETHICS/v9 [THIS IS THE MAIN]/Regional Cultural Persona/
  deepseek_deepseek-r1-distill-qwen-32b/<persona>/*.json

result/GMS8K/regional_cultural_persona/
  deepseek_deepseek-r1-distill-qwen-32b/<persona>/gsm8k_*.json
```

Output ditulis langsung di komputer Windows. Vast hanya menjalankan model; hasil eksperimen tidak perlu diunduh dari Vast.

### Dataset Coverage

- ETHICS: 1.514 soal per persona × 16 persona = 24.224 respons utama.
- GSM8K: 1.319 soal per persona × 16 persona = 21.104 respons utama.
- Total full run: 45.328 respons utama, ditambah maksimal 24 warmup.
- `neutral` tidak dijalankan karena scope ini hanya melengkapi 16 persona regional.

### Constraints

- BF16 tanpa quantization.
- Temperature tetap `0.0` agar konsisten dengan eksperimen repo.
- Maksimum output `4096` token.
- Satu endpoint vLLM dipakai paralel oleh ETHICS dan GSM8K.
- Default launcher: 8 request ETHICS + 8 request GSM8K.
- Full run tidak dapat dimulai tanpa flag keselamatan `--confirm_full`.

Catatan metodologi: model card DeepSeek merekomendasikan temperature `0.6` untuk penggunaan umum, tetapi repo ini mengunci `0.0` demi konsistensi eksperimen lintas model. Jangan mengubahnya di tengah eksperimen.

### Success Criteria

- `/v1/models` memuat alias `deepseek/deepseek-r1-distill-qwen-32b`.
- Smoke test menghasilkan 12 JSON sukses.
- Pilot selesai tanpa kegagalan API atau JSON rusak.
- Verifikasi akhir menunjukkan 1.514 ETHICS dan 1.319 GSM8K untuk setiap persona.

## 2. DESIGN PLAN

Alurnya sengaja dibagi menjadi gerbang murah. Vast tidak digunakan sebelum pemeriksaan lokal lulus. Sesudah instance hidup, SSH tunnel mengakses port internal vLLM secara langsung sehingga URL Cloudflare, cookie portal, dan `OPEN_BUTTON_TOKEN` tidak dibutuhkan runner.

- `plan`: validasi model, persona, dataset, command, dan jumlah call; nol API call.
- `preflight`: buka tunnel sementara dan cek `/v1/models`; nol generation call.
- `smoke`: dua persona, mencakup jalur explicit dan implicit; 12 call.
- `pilot`: seluruh 16 persona dalam sampel kecil; maksimal 152 call.
- `full`: seluruh 45.328 respons utama; harus memakai `--confirm_full`.
- `verify`: audit file lokal; tidak membutuhkan Vast.

## 3. IMPLEMENTATION — LANGKAH YANG HARUS DIIKUTI

Semua command lokal dijalankan dari PowerShell di:

```powershell
cd D:\TugasAkhir
```

### Langkah 0 — Validasi repo sebelum menyewa

Jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py --mode plan
```

Harus muncul:

```text
[LOCAL] model order 43: correct
[LOCAL] regional personas: 16/16
[LOCAL] ETHICS samples: 1514
[LOCAL] GSM8K samples: 1319
Plan complete: local inputs are valid
```

Jika command ini gagal, jangan rent GPU.

### Langkah 1 — Siapkan SSH key sebelum menyewa

**Status: sudah selesai.** Key khusus Vast sudah dibuat dan sudah terdaftar di akun:

```text
private : C:\Users\ASUS\.ssh\id_ed25519_vast
public  : C:\Users\ASUS\.ssh\id_ed25519_vast.pub
comment : vast-tugasakhir
```

Key ini sengaja dibuat **tanpa passphrase** karena launcher menjalankan SSH dengan
`BatchMode=yes`; key ber-passphrase akan menggagalkan tunnel tanpa sempat bertanya.
Key ini juga terpisah dari `id_ed25519_personal` dan `id_ed25519_samuel` yang dipakai
untuk GitHub, sehingga bisa dicabut dari Vast tanpa mengganggu akses lain.

Karena namanya bukan nama default, **setiap** command yang memakai SSH wajib menambahkan:

```powershell
--ssh_identity_file "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Jika suatu saat key perlu dibuat ulang:

```powershell
ssh-keygen -t ed25519 -C "vast-tugasakhir" -f "$env:USERPROFILE\.ssh\id_ed25519_vast"
```

Jangan pernah membagikan file privat `id_ed25519_vast`; yang boleh dimasukkan ke Vast
hanya isi file `.pub`. Di Vast.ai buka **Account/Keys → SSH Keys → + New**, lalu paste.
Lakukan ini sebelum membuat instance karena key baru tidak otomatis masuk ke instance lama.

### Langkah 2 — Buat template vLLM yang benar sebelum menyewa

**Status: sudah selesai.** Template sudah tersimpan di **My Templates** sebagai:

```text
vLLM DeepSeek R1 Distill Qwen 32B (TA)   (Private)
image             : vastai/vllm:v0.26.0-cuda-13.0
container disk    : 150 GB
max-num-seqs      : 128   <-- WAJIB, lihat catatan di bawah
```

**`--max-num-seqs 128` adalah setelan paling berdampak di seluruh runbook ini.**
Template bawaan Vast memakai `16`, dan itu membatasi throughput ke ~355 token/detik.
Pada 128 hasilnya ~1.900-2.100 token/detik — **naik sekitar 5,5x** dengan biaya sewa
yang sama persis. KV cache pada 128 terpakai 47-75%, jadi masih aman; jangan naikkan
ke 256 karena vLLM melaporkan `Maximum concurrency for 8,192 tokens: 12.25x` dan
KV akan penuh sehingga terjadi preemption.

Setelan ini **harus ada di `VLLM_ARGS` template sebelum rent**. Menambalnya pada
instance yang sudah jalan (lewat `/etc/vllm-args.conf` + `supervisorctl restart`)
tidak bekerja: supervisor melaporkan "started" tetapi proses vLLM lama tidak pernah
mati. Membunuhnya paksa juga berisiko — `pkill -f "vllm serve"` akan mencocokkan
teks perintahnya sendiri dan membunuh sesi SSH sebelum sempat menghabisi
`VLLM::EngineCore`, yang lalu menahan seluruh VRAM sehingga setiap start berikutnya
gagal OOM. Kalau terpaksa melakukannya, bunuh per-PID dari
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` lalu pastikan
`nvidia-smi` menunjukkan VRAM kosong sebelum start ulang.

Container disk dinaikkan dari default 24 GB karena checkpoint BF16 32B berukuran
puluhan GB dan diunduh ke `/workspace/models` di disk container. Disk 24 GB akan
gagal saat download, bukan saat startup.

Langkah di bawah hanya untuk referensi bila template perlu dibuat ulang.

1. Buka **Vast.ai → Templates**.
2. Cari template resmi/recommended **vLLM**.
3. Klik ikon pensil untuk membuat salinan di **My Templates**.
4. Jangan mengganti image bawaan template.
5. Ubah `VLLM_MODEL` menjadi:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

6. Ubah `VLLM_ARGS` menjadi satu baris berikut:

```text
--host 127.0.0.1 --port 18000 --served-model-name deepseek/deepseek-r1-distill-qwen-32b --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --enable-prefix-caching --reasoning-parser deepseek_r1 --download-dir /workspace/models
```

7. Simpan template.

Alasan memakai port `18000`: template recommended Vast menjalankan vLLM pada port internal ini, lalu portal/Caddy mem-proxy port publik `8000`. Jangan menjalankan server vLLM kedua secara manual pada port `8000` di dalam instance.

### Langkah 3 — Pilih mesin

Gunakan filter berikut:

- On-Demand.
- Verified host.
- Reliability minimal 99%.
- Tepat 1 GPU dengan VRAM 96 GB; pilihan aman: RTX PRO 6000 Blackwell 96 GB.
- CPU RAM minimal 128 GB.
- Disk minimal 150 GB.
- Download minimal 500 Mbps; lebih tinggi lebih baik karena checkpoint BF16 sekitar puluhan GB.
- Direct SSH tersedia.

Untuk konfigurasi default 16 sequence, jangan pilih GPU 48 GB. GPU 80 GB dapat terlalu sempit untuk BF16 + KV cache; gunakan 96 GB agar tidak membuang waktu karena OOM.

Sebelum menekan **Rent**, catat harga per jam. Perkiraan biaya minimum startup adalah:

```text
harga_per_jam × menit_download_model / 60
```

### Langkah 4 — Tunggu model, bukan hanya status “Running”

1. Setelah instance berstatus **Running**, klik **Logs** pada kartu instance.
2. Template akan mengunduh model dan memulai `vllm serve` otomatis.
3. Tunggu sampai log menunjukkan API server siap menerima request.

Status `Running` hanya berarti container hidup. Status itu belum membuktikan checkpoint selesai diunduh atau endpoint model siap.

URL seperti `https://...trycloudflare.com/#/tools` adalah halaman bantuan Instance Portal. Runner repo tidak memakai URL itu. Respons `401 Unauthorized` dari URL tersebut berarti portal meminta autentikasi, bukan berarti GPU mati.

### Langkah 5 — Ambil dua data dari tombol SSH

Klik ikon **SSH** pada kartu instance. Vast akan menampilkan command yang bentuknya seperti:

```text
ssh -p 20544 root@142.214.185.187
```

Ambil:

- SSH port: `20544`.
- SSH target: `root@142.214.185.187`.

Jangan mengirim `OPEN_BUTTON_TOKEN`, API key, atau private key kepada siapa pun.

### Langkah 6 — Preflight tanpa generation call

Ganti contoh target dan port dengan milik instance:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py `
  --mode preflight `
  --ssh_target "root@142.214.185.187" `
  --ssh_port 20544
```

Launcher otomatis melakukan hal berikut:

1. Membuka SSH tunnel `localhost:8000 → Vast:18000`.
2. Memanggil `/v1/models`.
3. Menolak model default 8B atau alias yang salah.
4. Menutup tunnel kembali.

Harus berakhir dengan:

```text
[PREFLIGHT] SSH tunnel: ready
Preflight complete: model is ready; no generation calls or files were written.
```

Jika private key tidak berada di lokasi default:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py `
  --mode preflight `
  --ssh_target "root@142.214.185.187" `
  --ssh_port 20544 `
  --ssh_identity_file "C:\path\to\private_key"
```

### Langkah 7 — Smoke test: hanya 12 call

Setelah preflight lulus:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py `
  --mode smoke `
  --ssh_target "root@142.214.185.187" `
  --ssh_port 20544
```

Smoke test memakai:

- `north_american_explicit`.
- `north_american_implicit_1`.
- 1 soal dari masing-masing 4 subset ETHICS: 8 respons utama.
- 1 soal GSM8K per persona: 2 respons utama.
- 1 warmup implicit per runner: 2 respons.
- Total: 12 call.

Cek jumlah JSON:

```powershell
$ethicsSmoke = "result\ETHICS\v9 [THIS IS THE MAIN]\Regional Cultural Persona\deepseek_deepseek-r1-distill-qwen-32b"
$gsmSmoke = "result\GMS8K\regional_cultural_persona\deepseek_deepseek-r1-distill-qwen-32b"

(Get-ChildItem -LiteralPath $ethicsSmoke -Recurse -Filter "*.json").Count
(Get-ChildItem -LiteralPath $gsmSmoke -Recurse -Filter "*.json").Count
```

Harapan setelah smoke pertama:

```text
9
3
```

Cari kegagalan teknis:

```powershell
Get-ChildItem -LiteralPath $ethicsSmoke,$gsmSmoke -Recurse -Filter "*.json" | ForEach-Object {
  try {
    $j = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    if ($j.response.success -ne $true) { $_.FullName }
  } catch {
    $_.FullName
  }
}
```

Tidak boleh ada path yang keluar. Buka minimal satu JSON explicit, satu JSON implicit, dan satu GSM8K untuk memeriksa response serta metadata model.

### Langkah 8 — Pilot seluruh persona

Hanya jalankan bila smoke bersih:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py `
  --mode pilot `
  --ssh_target "root@142.214.185.187" `
  --ssh_port 20544
```

Pilot menargetkan maksimal 152 call. File smoke yang sudah sukses otomatis dilewati, sehingga call baru akan lebih sedikit.

Catat:

- waktu mulai dan selesai pilot;
- saldo sebelum dan sesudah pilot;
- jumlah response gagal;
- token dan latency pada beberapa JSON;
- pemakaian VRAM dari Vast logs atau `nvidia-smi`.

Estimasi kasar durasi full dari pilot:

```text
estimasi_jam_full ≈ jam_pilot × (45.328 / 128 respons utama pilot)
```

Ini hanya perkiraan karena panjang reasoning berbeda antarsoal. Jangan menjalankan full sebelum perkiraan waktu dan biayanya masuk akal.

### Langkah 9 — Full run

Full hanya bisa dimulai dengan acknowledgement eksplisit:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py `
  --mode full `
  --confirm_full `
  --ssh_target "root@142.214.185.187" `
  --ssh_port 20544
```

Launcher menjalankan ETHICS dan GSM8K paralel terhadap endpoint yang sama. `--skip_existing` selalu aktif. Jika listrik, internet, SSH, atau proses terputus, jalankan command full yang sama; hanya JSON sukses yang dilewati, sedangkan JSON gagal/rusak dijalankan ulang.

Menekan `Ctrl+C` menghentikan kedua runner lokal dan menutup tunnel sementara.

### Langkah 10 — Verifikasi hasil lokal

Verifikasi tidak membutuhkan Vast atau SSH:

```powershell
.\.venv\Scripts\python.exe scripts\run_deepseek_regional_vast.py --mode verify
```

Hasil lengkap harus menunjukkan:

- 1.514 JSON ETHICS pada setiap persona.
- 1.319 JSON GSM8K pada setiap persona.
- Tidak ada `response.success = false`.
- Total 24.224 respons ETHICS dan 21.104 respons GSM8K.

`parsing_failed` ETHICS dilaporkan terpisah; API dapat sukses walaupun label akhir perlu reparsing.

### Langkah 11 — Hentikan biaya

Setelah verifikasi berhasil:

1. Pastikan output benar-benar ada di `D:\TugasAkhir\result`.
2. Di Vast, pilih **Destroy/Delete instance**.
3. Pastikan tidak ada volume terpisah yang masih ditagih jika memang tidak dibutuhkan.

`Stop` menghentikan biaya GPU tetapi storage instance masih dapat ditagih. `Destroy` mengakhiri instance dan menghapus disk container. Karena output eksperimen disimpan lokal, hasil repo tetap aman.

## 3b. THROUGHPUT — DUA TUAS YANG TERBUKTI

Diukur pada sesi 10 Agustus 2026, RTX PRO 6000 96 GB, $1,153/jam.

| Konfigurasi | Respons/jam |
|---|---|
| `max-num-seqs 16`, 1 launcher (8+8) | ~1.500 |
| `max-num-seqs 128`, 1 launcher (110+110) | ~5.500 |
| `max-num-seqs 128`, 4 runner terpisah per blok regional | **~11.000** |

**Tuas 1 — `--max-num-seqs 128` di template.** Lihat Langkah 2.

**Tuas 2 — pecah runner per blok regional.** Satu runner memproses persona secara
berurutan, sehingga setiap pergantian persona meninggalkan lembah slot kosong
(`running` sempat jatuh ke 61-85 dari 128 dengan `waiting: 0`). Menjalankan empat
runner pada set persona yang **disjoint** membuat lembah satu runner tertutup oleh
runner lain:

```powershell
# empat proses paralel, masing-masing --concurrency 45
--personas north_american_explicit,north_american_implicit_1,...
--personas east_asian_explicit,...
--personas south_asian_explicit,...
--personas sub_saharan_african_explicit,...
```

Set persona wajib disjoint. Dua runner pada persona yang sama akan menembak soal
yang sama dua kali karena `--skip_existing` hanya memeriksa file saat tugas dimulai.

**Cara membaca siapa yang jadi pembatas**, dari `/metrics`:

| Gejala | Pembatas | Tindakan |
|---|---|---|
| `running < max-num-seqs`, `waiting = 0` | klien | tambah runner / concurrency |
| `running = max-num-seqs`, `waiting > 0` | server | naikkan `max-num-seqs` |

**Jangan memakai jumlah proses sebagai sinyal selesai.** `.venv\Scripts\python.exe`
adalah shim yang memunculkan `C:\Python314\python.exe` sebagai anak, sehingga satu
runner logis tampak sebagai dua proses dan `wait` dapat kembali sebelum pekerjaan
benar-benar selesai. Gunakan **hitungan file per persona** sebagai satu-satunya
sinyal kemajuan.

**Naikkan target secara bertangga** (60 → 200 → 400 → 600 → ...), bukan langsung ke
target akhir. Total pekerjaannya identik, tetapi tangga menjamin ke-16 persona
berada di level yang sama bila credit habis di tengah jalan. Syarat naik tangga
adalah **nilai minimum** per-persona mencapai target — dan jangan memakai nilai
minimum itu sebagai deteksi macet, karena ia baru bergerak di akhir.

## 4. CHECKLIST

- [x] `--mode plan` lulus sebelum rent.
- [x] SSH public key sudah ditambahkan sebelum instance dibuat.
- [x] `VLLM_MODEL` menunjuk checkpoint DeepSeek Qwen 32B.
- [x] `VLLM_ARGS` memakai port `18000` dan alias repo yang tepat.
- [ ] Mesin memiliki 1 GPU 96 GB dan disk minimal 150 GB.
- [ ] `--mode preflight` lulus tanpa generation call.
- [ ] Smoke menghasilkan 9 JSON ETHICS + 3 JSON GSM8K tanpa failure.
- [ ] Pilot selesai dan biaya full sudah diperkirakan.
- [ ] Full dijalankan dengan `--confirm_full`.
- [ ] `--mode verify` lulus sebelum instance dihancurkan.
- [ ] Instance/volume yang tidak dipakai sudah dihapus.

## 5. TESTING DAN TROUBLESHOOTING

| Error | Arti | Tindakan |
|---|---|---|
| Portal `401 Unauthorized` | URL Cloudflare dilindungi autentikasi | Abaikan portal untuk runner; gunakan SSH tunnel otomatis. |
| `Permission denied (publickey)` | SSH key tidak masuk ke instance | Pastikan key ditambahkan sebelum rent; cek target, port, dan `--ssh_identity_file`. |
| `SSH tunnel exited with code ...` | SSH target/port/key salah | Salin ulang command dari tombol SSH; jangan lanjut ke smoke. |
| `vLLM was not ready` | SSH hidup, model masih download/startup atau crash | Buka Instance Logs; tunggu readiness atau periksa OOM/disk. |
| `serves [...], not deepseek/...32b` | Template menjalankan model/alias salah | Perbaiki `VLLM_MODEL` dan `--served-model-name`. |
| `Address already in use` pada local 8000 | Ada tunnel/service lama | Tutup proses lama atau tambahkan `--local_tunnel_port 8002`. |
| CUDA OOM | VRAM tidak cukup untuk konfigurasi | Gunakan GPU 96 GB; jangan memulai full pada mesin yang gagal smoke/pilot. |
| Tunnel putus saat full | Jaringan/SSH terputus | Jalankan ulang command full; output sukses akan di-resume. |

Jika template ternyata memakai port internal selain `18000`, tambahkan:

```powershell
--remote_vllm_port PORT_YANG_BENAR
```

## 6. INTEGRATION

Launcher mempertahankan struktur output canonical repo dan tidak mengubah prompt, temperature, schema JSON, atau persona. ETHICS serta GSM8K berjalan melalui runner yang sudah ada; launcher hanya memvalidasi kontrak lokal, mengelola tunnel, memeriksa model, dan mengatur urutan smoke/pilot/full.

Hasil per-question dapat diteruskan ke pipeline rebuild, parsing, accuracy, dan analisis persona yang sudah ada. Full run bersifat resumable karena file hanya dianggap selesai jika JSON dapat dibaca dan `response.success` bernilai `true`.
