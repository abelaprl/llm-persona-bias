# Cara Menjalankan — GSM8K Regional Cultural Persona

Runner: `src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py`
Persona file: `inputs/regional_cultural_persona.json`
Output: `result/GMS8K/regional_cultural_persona/{model_slug}/{persona}/gsm8k_NNNNN.json`

> Jalankan semua perintah dari folder `D:\TugasAkhir`.

---

## Daftar Model (pilih berdasarkan `order`)

| order | Nama Model | model_id |
|------:|-----------|---------|
| 0  | Groq: Llama 3.1 8B Instant | groq/llama-3.1-8b-instant |
| 4  | DeepSeek V3.1 (free) | deepseek/deepseek-chat-v3.1:free |
| 5  | OpenAI: gpt-oss-20b (free) | openai/gpt-oss-20b:free |
| 9  | Meta: Llama 3.3 8B Instruct (free) | meta-llama/llama-3.3-8b-instruct:free |
| 11 | Qwen: Qwen3 235B A22B (free) | qwen/qwen3-235b-a22b:free |
| 12 | Google: Gemini 2.0 Flash (free) | google/gemini-2-0-flash:free |
| 14 | Meta: Llama 3.3 70B Instruct (free) | meta-llama/llama-3.3-70b-instruct:free |
| 17 | Mistral: Mistral Small 3.1 24B (free) | mistralai/mistral-small-3.1-24b-instruct:free |
| 19 | TNG: R1T Chimera (free) | tngtech/tng-r1t-chimera:free |
| 22 | Google: Gemma 3 27B (free) | google/gemma-3-27b-it:free |
| 23 | Google: Gemma 3 12B (free) | google/gemma-3-12b-it:free |
| 24 | Meta: Llama 3.1 405B Instruct (free) | meta-llama/llama-3.1-405b-instruct:free |
| 25 | Meta: Llama 3.3 70B Instruct (free) | meta-llama/llama-3.3-70b-instruct:free |
| 31 | OpenAI: GPT-OSS-20B | openai/gpt-oss-20b |
| 32 | Qwen: Qwen3 Next 80B A3B (Thinking) | qwen/qwen3-next-80b-a3b-thinking |
| 33 | Qwen: Qwen3 235B A22B Instruct 2507 | qwen/qwen3-235b-a22b-2507 |
| 34 | OpenAI: GPT-OSS-120B | openai/gpt-oss-120b |
| 35 | Meta: Llama 3.3 70B Instruct | meta-llama/llama-3.3-70b-instruct |
| 36 | Google: Gemma 3 27B | google/gemma-3-27b-it |
| 37 | Google: Gemma 2 9B | google/gemma-2-9b-it |
| 38 | DeepSeek V3.2 | deepseek/deepseek-v3.2 |
| 39 | Google: Gemini 2.5 Flash | google/gemini-2.5-flash |
| 40 | Meta: Llama 3.1 8B Instruct | meta-llama/llama-3.1-8b-instruct |
| 41 | Mistral: Ministral 8B | mistralai/ministral-8b-2512 |
| 42 | Allen AI: OLMo 3.1 32B Instruct | allenai/olmo-3.1-32b-instruct |
| **43** | **DeepSeek: R1 Distill Qwen 32B** | deepseek/deepseek-r1-distill-qwen-32b |
| 44 | Arcee AI: Trinity Large Thinking | arcee-ai/trinity-large-thinking |
| 45 | MoonshotAI: Kimi K2 Thinking | moonshotai/kimi-k2-thinking |
| 46 | NVIDIA: Nemotron Nano 9B V2 | nvidia/nemotron-nano-9b-v2 |
| 47 | Qwen: Qwen3 8B | qwen/qwen3-8b |

Ganti angka `--model_order` di perintah di bawah untuk memilih model yang diinginkan.

---

## Daftar Persona Regional (16 total)

| Persona | Tipe | Keterangan |
|---------|------|-----------|
| `north_american_explicit` | Explicit | "I am from North America." |
| `north_american_implicit_1` | Implicit | A — proksi demografis (keputusan individual) |
| `north_american_implicit_2` | Implicit | B — proksi praktik (Thanksgiving / turkey) |
| `north_american_implicit_3` | Implicit | C — proksi life event (wedding registry) |
| `east_asian_explicit` | Explicit | "I am from East Asia." |
| `east_asian_implicit_1` | Implicit | A — proksi demografis (karier vs keluarga) |
| `east_asian_implicit_2` | Implicit | B — proksi praktik (Chūnjié / hongbao) |
| `east_asian_implicit_3` | Implicit | C — proksi life event (tea ceremony) |
| `south_asian_explicit` | Explicit | "I am from South Asia." |
| `south_asian_implicit_1` | Implicit | A — proksi demografis (rahasia keluarga) |
| `south_asian_implicit_2` | Implicit | B — proksi praktik (Diwali / diyas) |
| `south_asian_implicit_3` | Implicit | C — proksi life event (mehendi ceremony) |
| `sub_saharan_african_explicit` | Explicit | "I am from Sub-Saharan Africa." |
| `sub_saharan_african_implicit_1` | Implicit | A — proksi demografis (mendamaikan teman) |
| `sub_saharan_african_implicit_2` | Implicit | B — proksi praktik (harvest celebration) |
| `sub_saharan_african_implicit_3` | Implicit | C — proksi life event (matching fabric wedding) |

---

## 0. (Opsional) Aktifkan venv

```powershell
.venv\Scripts\activate
```

---

## ⚠️ Catatan paste di PowerShell

Jangan paste banyak baris perintah `python ...` sekaligus — baris berikutnya bisa nempel ke perintah sebelumnya dan memicu error `unrecognized arguments: python`.
Pilih salah satu cara di bawah.

---

## Cara 1 — Loop sekuensial (disarankan)

Ganti `$modelOrder = 31` dengan nomor order model yang kamu pilih dari tabel di atas.
Paste **sekali**; tiap persona jalan **berurutan** (yang berikutnya mulai setelah yang sekarang selesai).

```powershell
$modelOrder = 31   # <-- ganti nomor model di sini

$personas = @(
  "north_american_explicit","north_american_implicit_1","north_american_implicit_2","north_american_implicit_3",
  "east_asian_explicit","east_asian_implicit_1","east_asian_implicit_2","east_asian_implicit_3",
  "south_asian_explicit","south_asian_implicit_1","south_asian_implicit_2","south_asian_implicit_3",
  "sub_saharan_african_explicit","sub_saharan_african_implicit_1","sub_saharan_african_implicit_2","sub_saharan_african_implicit_3"
)
foreach ($p in $personas) {
  Write-Host "`n===== $p =====" -ForegroundColor Cyan
  python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py `
    --model_order $modelOrder `
    --personas $p `
    --limit_samples 1319 `
    --skip_existing
}
```

> `--limit_samples 1319` = jalankan 1319 soal pertama GSM8K (subset standar penelitian ini). Set ke `0` untuk semua 8819 soal.

---

## Cara 2 — Manual, satu per satu

Copy **satu baris**, Enter, tunggu sampai muncul `Done.`, baru lanjut baris berikutnya.
Ganti `--model_order 31` dengan nomor model yang dipilih.

```powershell
# NORTH AMERICAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas north_american_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas north_american_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas north_american_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas north_american_implicit_3 --limit_samples 1319 --skip_existing

# EAST ASIAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas east_asian_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas east_asian_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas east_asian_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas east_asian_implicit_3 --limit_samples 1319 --skip_existing

# SOUTH ASIAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas south_asian_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas south_asian_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas south_asian_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas south_asian_implicit_3 --limit_samples 1319 --skip_existing

# SUB-SAHARAN AFRICAN
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas sub_saharan_african_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas sub_saharan_african_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas sub_saharan_african_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --personas sub_saharan_african_implicit_3 --limit_samples 1319 --skip_existing
```

---

## Cara 3 — Semua persona sekaligus (satu perintah)

Tanpa `--personas`, runner menjalankan **semua 16 persona** dari file.

```powershell
python src/gsm8k_runner/run_gsm8k_regional_cultural_personas.py --model_order 31 --limit_samples 1319 --skip_existing
```

> Warmup implicit persona dilakukan di awal (step 1/2), baru dilanjutkan run per soal (step 2/2). Jika warmup gagal, runner berhenti otomatis untuk mencegah cascading FAIL.

---

## Parameter Lengkap

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| `--model_order` | `43` | Nomor order model dari `config/model_keys.json` |
| `--limit_samples` | `1` | Jumlah soal GSM8K yang dijalankan (`0` = semua 8819) |
| `--personas` | *(semua)* | Persona spesifik, pisah koma: `north_american_explicit,east_asian_implicit_1` |
| `--persona_file` | `inputs/regional_cultural_persona.json` | Path ke file persona |
| `--skip_existing` | `False` | Lewati soal yang file JSON-nya sudah ada |
| `--force_warmup` | `False` | Paksa regenerasi warmup meski sudah ada |
| `--output_root` | `result/GMS8K/regional_cultural_persona` | Folder output |
| `--sleep_sec` | `1.0` | Jeda antar request API (detik) |

---

## Struktur Output

```
result/GMS8K/regional_cultural_persona/
└── {model_slug}/                       # e.g. deepseek_deepseek-r1-distill-qwen-32b
    ├── north_american_explicit/
    │   ├── gsm8k_00001.json
    │   └── ...
    ├── north_american_implicit_1/
    │   ├── warmup/
    │   │   └── persona_warmup.json
    │   ├── gsm8k_00001.json
    │   └── ...
    └── ... (semua 16 persona)
```

---

## Setelah Selesai — Rebuild CSV

```powershell
python src/extract/rebuild_results_csv_root.py
```

Lalu update laporan Excel:

```powershell
python src/reports/generate_results_csv.py
```
