# Cara Menjalankan — GSM8K Religion Persona

Runner: `src/gsm8k_runner/run_gsm8k_religion_personas.py`
Output: `result/GMS8K/Religion_Persona/{model_slug}/{persona}/gsm8k_NNNNN.json`

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

## Daftar Persona Religion (16 total)

| Persona | Tipe | Keterangan |
|---------|------|-----------|
| `islam_explicit` | Explicit | "I am a Muslim." |
| `islam_implicit_1` | Implicit | A — membaca Al-Quran / kitab |
| `islam_implicit_2` | Implicit | B — Ramadan / hari raya |
| `islam_implicit_3` | Implicit | C — shalat & masjid |
| `christian_explicit` | Explicit | "I am a Christian." |
| `christian_implicit_1` | Implicit | A — membaca Alkitab |
| `christian_implicit_2` | Implicit | B — Christmas & Easter / Lent |
| `christian_implicit_3` | Implicit | C — Sunday service |
| `jewish_explicit` | Explicit | "I am Jewish." |
| `jewish_implicit_1` | Implicit | A — Torah & rabbi |
| `jewish_implicit_2` | Implicit | B — Yom Kippur |
| `jewish_implicit_3` | Implicit | C — synagogue & Shabbat |
| `atheist_explicit` | Explicit | "I am an atheist." |
| `atheist_implicit_1` | Implicit | A — tanpa kitab suci |
| `atheist_implicit_2` | Implicit | B — tanpa kalender religius |
| `atheist_implicit_3` | Implicit | C — tanpa komunitas religius |

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

Ganti `$modelOrder = 43` dengan nomor order model yang kamu pilih dari tabel di atas.
Paste **sekali**; tiap persona jalan **berurutan** (yang berikutnya mulai setelah yang sekarang selesai).

```powershell
$modelOrder = 43   # <-- ganti nomor model di sini

$personas = @(
  "islam_explicit","islam_implicit_1","islam_implicit_2","islam_implicit_3",
  "christian_explicit","christian_implicit_1","christian_implicit_2","christian_implicit_3",
  "jewish_explicit","jewish_implicit_1","jewish_implicit_2","jewish_implicit_3",
  "atheist_explicit","atheist_implicit_1","atheist_implicit_2","atheist_implicit_3"
)
foreach ($p in $personas) {
  Write-Host "`n===== $p =====" -ForegroundColor Cyan
  python src/gsm8k_runner/run_gsm8k_religion_personas.py `
    --model_order $modelOrder `
    --personas $p `
    --limit_samples 100 `
    --skip_existing
}
```

> `--limit_samples 100` artinya ambil 100 soal pertama dari GSM8K. Hapus parameter ini (atau set ke `0`) untuk menjalankan semua 8819 soal.

---

## Cara 2 — Manual, satu per satu

Copy **satu baris**, Enter, tunggu sampai muncul `Done.`, baru lanjut baris berikutnya.
Ganti `--model_order 43` dengan nomor model yang dipilih.

```powershell
# ISLAM
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas islam_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas islam_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas islam_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas islam_implicit_3 --limit_samples 1319 --skip_existing

# CHRISTIAN
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas christian_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas christian_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas christian_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas christian_implicit_3 --limit_samples 1319 --skip_existing

# JEWISH
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas jewish_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas jewish_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas jewish_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas jewish_implicit_3 --limit_samples 1319 --skip_existing

# ATHEIST
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas atheist_explicit --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas atheist_implicit_1 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas atheist_implicit_2 --limit_samples 1319 --skip_existing
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 41 --personas atheist_implicit_3 --limit_samples 1319 --skip_existing
```

---

## Cara 3 — Semua persona sekaligus (satu perintah)

Runner mendukung semua persona dijalankan dalam satu invokasi tanpa `--personas` (default: semua persona dari file).

```powershell
python src/gsm8k_runner/run_gsm8k_religion_personas.py --model_order 43 --limit_samples 100 --skip_existing
```

> Warmup implicit persona dilakukan di awal (step 1/2), baru dilanjutkan run per soal (step 2/2). Jika warmup gagal, runner berhenti otomatis untuk mencegah cascading FAIL.

---

## Parameter Lengkap

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| `--model_order` | `43` | Nomor order model dari `config/model_keys.json` |
| `--limit_samples` | `1` | Jumlah soal GSM8K yang dijalankan (`0` = semua) |
| `--personas` | *(semua)* | Persona spesifik, pisah koma: `islam_explicit,atheist_implicit_1` |
| `--persona_file` | `inputs/religion_persona.json` | Path ke file persona |
| `--skip_existing` | `False` | Lewati soal yang file JSON-nya sudah ada |
| `--force_warmup` | `False` | Paksa regenerasi warmup meski sudah ada |
| `--output_root` | `result/GMS8K/Religion_Persona` | Folder output |
| `--sleep_sec` | `1.0` | Jeda antar request API (detik) |

---

## Struktur Output

```
result/GMS8K/Religion_Persona/
└── {model_slug}/              # e.g. deepseek_deepseek-r1-distill-qwen-32b
    ├── islam_explicit/
    │   ├── gsm8k_00001.json
    │   ├── gsm8k_00002.json
    │   └── ...
    ├── islam_implicit_1/
    │   ├── warmup/
    │   │   └── persona_warmup.json
    │   ├── gsm8k_00001.json
    │   └── ...
    └── ... (semua 16 persona)
```

---

## Setelah Selesai — Rebuild CSV

Setelah semua persona selesai, rebuild aggregated CSV:

```powershell
python src/extract/rebuild_results_csv_root.py
```

Lalu update laporan Excel:

```powershell
python src/reports/generate_results_csv.py
```
