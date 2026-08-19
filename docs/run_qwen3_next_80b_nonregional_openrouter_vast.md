# Rerun Qwen3 Next 80B melalui OpenRouter di Vast.ai

## Ruang lingkup

Runner ini memanggil model OpenRouter berikut:

```text
qwen/qwen3-next-80b-a3b-thinking
https://openrouter.ai/api/v1/chat/completions
```

Vast.ai hanya menjadi komputer yang menjalankan Python. Model tidak dimuat ke
GPU Vast, sehingga RTX PRO 6000 tidak digunakan untuk inferensi ini.

Tersedia dua manifest yang sengaja dipisahkan:

| Scope | ETHICS | GSM8K | Total |
|---|---:|---:|---:|
| Non-regional | 2.949 | 5.341 | 8.290 |
| Regional Cultural | 5.085 | 546 | 5.631 |
| **Gabungan** | **8.034** | **5.887** | **13.921** |

Komposisi masalah: 7.048 HTTP 403, 1.160 TLS, 56 DNS, 16 remote reset,
6 host-aborted, 2 JSON kosong, 1 JSON terpotong, dan 1 JSON hilang. Regional
Cultural tidak ada di manifest non-regional dan scope lawan selalu ditolak oleh
validasi runner.

Manifest Regional Cultural berisi 5.082 HTTP 403, 485 TLS, 51 DNS, 11
host-aborted, dan 2 remote-reset. Tidak ada HTTP 404, JSON invalid, atau JSON
hilang pada scope regional.

## File lama tidak perlu dihapus

Jangan hapus JSON error sebelum rerun. Runner melakukan hal berikut:

1. membaca hanya target yang tercantum di manifest;
2. melewati target yang sudah berisi respons sukses saat resume;
3. membackup file lama tepat sebelum target dipanggil;
4. menyimpan respons gagal/kosong ke folder attempts;
5. mengganti atau membuat JSON utama secara atomik hanya jika final assistant
   content tidak kosong.

Backup tersimpan di:

```text
result/_rerun_backup/qwen3_next_80b_nonregional/<run_id>/
```

Percobaan yang gagal tersimpan di:

```text
result/_rerun_attempts/qwen3_next_80b_nonregional/<run_id>/
```

## Persiapan di instance Vast.ai

Masuk ke folder repo atau hasil ekstraksi bundle, lalu pasang dependensi:

```bash
cd /workspace/TugasAkhir
python -m pip install -r requirements-vast-rerun.txt
```

Bundle lokal yang dapat diunggah ke Vast:

```text
dist/qwen3_next_80b_nonregional_openrouter_rerun.zip
dist/qwen3_next_80b_regional_openrouter_rerun.zip
```

Masukkan API key sebagai environment secret. Jangan menaruh key di Markdown,
ZIP, commit, atau command history:

```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
echo
```

Runner tetap mendukung fallback lokal `config/model_keys.json`, tetapi file itu
tidak disertakan dalam bundle dan tidak boleh di-push karena berisi secret.

## Urutan perintah yang disarankan

Validasi manifest tanpa API call dan tanpa menulis file:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py
```

Smoke test satu target terlebih dahulu:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --execute --limit 1 --concurrency 1
```

Lanjutkan batch kecil. Target yang sukses pada smoke test otomatis dilewati:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --execute --limit 10 --concurrency 2
```

Jika hasil batch kecil benar, jalankan seluruh target tersisa:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --execute --concurrency 4
```

Regional Cultural memakai runner yang sama tetapi manifest dan folder backup
terpisah. Validasi dahulu:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py --scope regional
```

### Health-check memakai skenario asli ETHICS v9 Regional Persona

Skenario pada `docs/run_ethics_v9_regional_persona.md` memakai runner eksperimen
asli. Gunakan skenario ini untuk memastikan model, key, prompt, dan parser bekerja
sebelum rerun manifest. Ini bukan pemilih file gagal.

Perintah berikut mengambil dua soal dari masing-masing empat subset ETHICS untuk
satu persona explicit, sehingga ada tepat 8 API call. Output sengaja diarahkan ke
folder smoke terpisah dan tidak menyentuh hasil utama:

```powershell
python ".\src\ethics_runner\run_ethics_prompt_v9.py" --model_order 32 --personas north_american_explicit --max_per_subset 2 --output_root "result\_smoke\qwen3_next_80b_regional" --json_only --skip_existing --concurrency 1
```

Di shell Linux/Vast.ai, gunakan bentuk satu baris ini:

```bash
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 32 --personas north_american_explicit --max_per_subset 2 --output_root 'result/_smoke/qwen3_next_80b_regional' --json_only --skip_existing --concurrency 1
```

Runner asli membutuhkan repo lengkap, termasuk data, input persona, dan
`config/model_keys.json` dengan key order 32 yang sudah dirotasi. Jangan arahkan
`--output_root` ke `result/ETHICS/v9 [THIS IS THE MAIN]`: runner asli tetap
menulis JSON kegagalan jika request ulang gagal.

Setelah 8 hasil smoke valid, kembali ke runner selektif di bawah. Runner selektif
hanya mengambil target pada manifest audit, membuat backup, mendukung resume,
dan tidak mengganti JSON sumber ketika respons baru gagal atau kosong.

Smoke test Regional Cultural:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --scope regional --execute --limit 1 --concurrency 1
```

Full Regional Cultural setelah smoke test berhasil:

```bash
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --scope regional --execute --concurrency 4
```

Menjalankan per dataset juga bisa:

```bash
# ETHICS saja: 2.949 target
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --dataset ethics --execute --concurrency 4

# GSM8K saja: 5.341 target
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --dataset gsm8k --execute --concurrency 4

# Regional ETHICS saja: 5.085 target
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --scope regional --dataset ethics --execute --concurrency 4

# Regional GSM8K saja: 546 target
python scripts/rerun_qwen3_next_80b_openrouter.py \
  --scope regional --dataset gsm8k --execute --concurrency 4
```

Gunakan `tmux` agar proses tidak berhenti saat koneksi SSH terputus. Perintah
full aman dijalankan kembali karena runner melakukan resume berdasarkan isi
JSON sukses, bukan dengan memindai model atau skenario lain.

## Batas OpenRouter

Sebanyak 7.048 kegagalan lama adalah HTTP 403 key-limit. Smoke test terbaru
menunjukkan endpoint dan model Qwen dapat diakses lagi, tetapi status key saat
audit hanya melaporkan `limit_remaining` sekitar 75,79. Nilai itu tidak boleh
dianggap otomatis cukup untuk 13.921 panggilan model thinking dengan budget
hingga 4.096 token. Jalankan bertahap dan periksa pemakaian OpenRouter sebelum
full rerun.

Respons yang menghabiskan token hanya untuk reasoning dan tidak menghasilkan
final content dianggap gagal; file JSON utama tidak akan ditimpa.

## File yang digunakan

- `scripts/rerun_qwen3_next_80b_openrouter.py`
- `scripts/rerun_qwen3_next_80b_nonregional_openrouter.py` (implementasi)
- `result/rerun_manifests/qwen3_next_80b_nonregional/qwen_nonregional_infra_all.csv`
- `result/rerun_manifests/qwen3_next_80b_regional/qwen_regional_infra_all.csv`
- `src/ethics_runner/run_ethics_prompt_v9.py`
- `requirements-vast-rerun.txt`

Tidak ada perintah rerun dalam dokumen ini yang dijalankan saat penyusunan.
