# Cara Menjalankan — GSM8K Regional Cultural Persona

Runner: `src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py`
Model default: order **43** (DeepSeek R1 Distill Qwen 32B), temperature **0.0**, max_tokens **4096**.
Output: `result/GMS8K/regional_cultural_persona/deepseek_deepseek-r1-distill-qwen-32b/<persona>/gsm8k_NNNNN.json`

> Jalankan semua perintah dari folder `D:\TugasAkhir`.

---

## 0. (Opsional) Aktifkan venv

```powershell
.venv\Scripts\activate
```

---

## Rerun yang kurang dari 1318

Hasil cek `result/GMS8K/regional_cultural_persona` menemukan 3 folder persona yang belum lengkap, semuanya untuk model `allenai_olmo-3.1-32b-instruct`.
Jalankan command berikut dari `D:\TugasAkhir`. Flag `--skip_existing` akan melewati file yang sudah ada dan hanya mengisi `gsm8k_*.json` yang missing sampai `--limit_samples 1319`.

```powershell
# allenai_olmo-3.1-32b-instruct/north_american_implicit_1
# Saat dicek: 1178 file, lanjut dari missing 1179..1319.
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 42 --personas north_american_implicit_1 --limit_samples 1319 --skip_existing

# allenai_olmo-3.1-32b-instruct/north_american_implicit_2
# Saat dicek: 249 file, lanjut dari missing 250..1319.
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 42 --personas north_american_implicit_2 --limit_samples 1319 --skip_existing

# allenai_olmo-3.1-32b-instruct/north_american_implicit_3
# Saat dicek: 42 file, lanjut dari missing 43..1319.
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 42 --personas north_american_implicit_3 --limit_samples 1319 --skip_existing
```

Ringkasan target:

| Model slug | model_order | Persona | Count saat dicek | Target |
|---|---:|---|---:|---:|
| `allenai_olmo-3.1-32b-instruct` | 42 | `north_american_implicit_1` | 1178 | 1319 |
| `allenai_olmo-3.1-32b-instruct` | 42 | `north_american_implicit_2` | 249 | 1319 |
| `allenai_olmo-3.1-32b-instruct` | 42 | `north_american_implicit_3` | 42 | 1319 |

---

## ⚠️ Catatan paste di PowerShell

Jangan paste banyak baris perintah `python ...` sekaligus — baris berikutnya bisa nempel ke perintah sebelumnya dan memicu error `unrecognized arguments: python`.
Pilih salah satu cara di bawah.

---

## Cara 1 — Loop sekuensial (disarankan)

Paste **sekali**; tiap persona jalan **berurutan** (yang berikutnya mulai setelah yang sekarang selesai).

```powershell
$personas = @(
  "western_explicit","western_implicit_1","western_implicit_2","western_implicit_3",
  "east_asian_explicit","east_asian_implicit_1","east_asian_implicit_2","east_asian_implicit_3",
  "south_asian_explicit","south_asian_implicit_1","south_asian_implicit_2","south_asian_implicit_3",
  "african_explicit","african_implicit_1","african_implicit_2","african_implicit_3"
)
foreach ($p in $personas) {
  Write-Host "`n===== $p =====" -ForegroundColor Cyan
  python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas $p --limit_samples 10 --skip_existing
}
```

---

## Cara 2 — Manual, satu per satu

Copy **satu baris**, Enter, tunggu sampai muncul `Done.`, baru lanjut baris berikutnya.

```powershell
# WESTERN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas western_explicit --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas western_implicit_1 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas western_implicit_2 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas western_implicit_3 --limit_samples 100 --skip_existing

# EAST ASIAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas east_asian_explicit --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas east_asian_implicit_1 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas east_asian_implicit_2 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas east_asian_implicit_3 --limit_samples 100 --skip_existing

# SOUTH ASIAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas south_asian_explicit --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas south_asian_implicit_1 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas south_asian_implicit_2 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas south_asian_implicit_3 --limit_samples 100 --skip_existing

# AFRICAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas african_explicit --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas african_implicit_1 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas african_implicit_2 --limit_samples 100 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --personas african_implicit_3 --limit_samples 100 --skip_existing
```

---

## Catatan

- Persona **implicit** otomatis membuat `warmup/persona_warmup.json` (1 call) sebelum 10 soal.
- `--skip_existing` membuat run **resume-able** dan tidak menimpa file yang sudah ada.
- 16 persona × 10 soal = **160** file `gsm8k_NNNNN.json` + warmup untuk 12 persona implicit.
- Tip anti salah ketik: ketik `python src/gsm8k_runner/run_gsm8k_` lalu tekan **Tab** untuk auto-complete nama file.

### Flag berguna
| Flag | Default | Fungsi |
|---|---|---|
| `--personas` | semua | Nama persona (pisah koma untuk beberapa) |
| `--limit_samples` | 1 | Jumlah soal GSM8K |
| `--skip_existing` | off | Lewati file yang sudah ada (resume) |
| `--model_order` | 43 | Pilih model dari `config/model_keys.json` |
| `--force_warmup` | off | Regenerate warmup walau sudah ada |
| `--output_root` | `result/GMS8K/regional_cultural_persona` | Folder output |
| `--sleep_sec` | 1.0 | Jeda antar call (detik) |

---

## Setelah selesai — cek kelengkapan

Hitung jumlah file & deteksi kegagalan:

```powershell
$root = "result/GMS8K/regional_cultural_persona/deepseek_deepseek-r1-distill-qwen-32b"
Get-ChildItem $root -Recurse -Filter "gsm8k_*.json" | Measure-Object | Select-Object -ExpandProperty Count
```

(Harapan: **160** file untuk 16 persona × 10 soal.)
