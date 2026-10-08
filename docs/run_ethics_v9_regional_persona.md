# Cara Menjalankan — ETHICS v9 Regional Cultural Persona

Runner: `src/ethics_runner/run_ethics_prompt_v9.py`
Persona file: `inputs/regional_cultural_persona.json` (sudah otomatis di-load runner)
Samples: `data/ethics/ETHICS_test_4subsets_10percent_from200.json` (1514 soal: commonsense + deontology + justice + virtue)

Setelah reorganisasi `data/`, tambahkan `--samples_file data/ethics/ETHICS_test_4subsets_10percent_from200.json` pada setiap perintah runner di bawah. Perubahan default runner menunggu izin sesuai AGENTS.md.
Output: `result/ETHICS/v9_pilot/Regional Cultural Persona/{model_slug}/{persona}/{sample_id}.json`

> Jalankan semua perintah dari folder `D:\TugasAkhir`.

---

## Daftar Model (pilih berdasarkan `order`)

ETHICS v9 dirancang untuk model pipeline (order **31–47**). Ganti `--model_order` dengan nomor di bawah.

| order | Nama Model | model_id |
|------:|-----------|---------|
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

---

## Daftar Persona Regional (16 + neutral)

`--group regional` otomatis menyertakan `neutral` sebagai kontrol + 16 persona regional.

| Persona | Tipe | Keterangan |
|---------|------|-----------|
| `neutral` | Control | Tanpa persona (kontrol) |
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

## 0b. (Opsional) Dry run — lihat prompt & parsing tanpa panggil API

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --parse_demo
```

## 0c. Health-check Qwen3 Next 80B (order 32) sebelum rerun

Gunakan runner skenario asli ini hanya untuk memastikan model, API key, prompt,
dan parser bekerja. Dua soal diambil dari masing-masing subset ETHICS untuk satu
persona explicit: 2 x 4 subset = 8 API call. Output diarahkan ke folder smoke
terpisah agar hasil utama tidak tersentuh.

```powershell
python ".\src\ethics_runner\run_ethics_prompt_v9.py" --model_order 32 --personas north_american_explicit --max_per_subset 2 --output_root "result\_smoke\qwen3_next_80b_regional" --json_only --skip_existing --concurrency 1
```

Versi Linux/Vast.ai:

```bash
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 32 --personas north_american_explicit --max_per_subset 2 --output_root 'result/_smoke/qwen3_next_80b_regional' --json_only --skip_existing --concurrency 1
```

Jalankan hanya setelah API key order 32 yang sempat terekspos sudah dicabut dan
diganti. Runner ini membaca key dari `config/model_keys.json`. Jangan mengarahkan
`--output_root` ke `result/ETHICS/v9 [THIS IS THE MAIN]`, karena runner eksperimen
asli tetap dapat menulis JSON berisi kegagalan ketika request gagal.

Setelah health-check berhasil, rerun hanya file audit yang gagal dengan panduan
`docs/run_qwen3_next_80b_nonregional_openrouter_vast.md`.

---

## ⚠️ Catatan paste di PowerShell

Jangan paste banyak baris perintah `python ...` sekaligus. Pilih salah satu cara di bawah.

---

## Cara 1 — Semua persona regional sekaligus (disarankan)

`--group regional` menjalankan neutral + 16 persona regional dalam satu invokasi.
Cukup ganti `--model_order`.

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --group regional --skip_existing
```

> Warmup untuk persona implicit dibuat otomatis (cache per persona). Output per-model CSV di-rebuild otomatis di akhir.

---

## Cara 2 — Loop sekuensial per persona

Ganti `$modelOrder = 31`. Paste **sekali**; tiap persona jalan berurutan.

```powershell
$modelOrder = 31   # <-- ganti nomor model di sini

$personas = @(
  "neutral",
  "north_american_explicit","north_american_implicit_1","north_american_implicit_2","north_american_implicit_3",
  "east_asian_explicit","east_asian_implicit_1","east_asian_implicit_2","east_asian_implicit_3",
  "south_asian_explicit","south_asian_implicit_1","south_asian_implicit_2","south_asian_implicit_3",
  "sub_saharan_african_explicit","sub_saharan_african_implicit_1","sub_saharan_african_implicit_2","sub_saharan_african_implicit_3"
)
foreach ($p in $personas) {
  Write-Host "`n===== $p =====" -ForegroundColor Cyan
  python src/ethics_runner/run_ethics_prompt_v9.py `
    --model_order $modelOrder `
    --personas $p `
    --skip_existing
}
```

---

## Cara 3 — Manual, satu per satu

Copy **satu baris**, Enter, tunggu sampai muncul `Done.`, baru lanjut. Ganti `--model_order 31`.

```powershell
# NEUTRAL (kontrol)
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas neutral --skip_existing

# NORTH AMERICAN
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas north_american_explicit --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas north_american_implicit_1 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas north_american_implicit_2 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas north_american_implicit_3 --skip_existing

# EAST ASIAN
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas east_asian_explicit --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas east_asian_implicit_1 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas east_asian_implicit_2 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas east_asian_implicit_3 --skip_existing

# SOUTH ASIAN
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas south_asian_explicit --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas south_asian_implicit_1 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas south_asian_implicit_2 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas south_asian_implicit_3 --skip_existing

# SUB-SAHARAN AFRICAN
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas sub_saharan_african_explicit --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas sub_saharan_african_implicit_1 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas sub_saharan_african_implicit_2 --skip_existing
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --personas sub_saharan_african_implicit_3 --skip_existing
```

---

## Cara 4 — Semua model pipeline (31–47) sekaligus

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --all_pipeline --group regional --skip_existing
```

---

## Parameter Lengkap

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| `--model_order` | *(wajib)* | Nomor order model dari `config/model_keys.json` |
| `--all_pipeline` | `False` | Jalankan semua model order 31–47 (abaikan `--model_order`) |
| `--group` | *(kosong)* | Grup persona: `regional`, `religion`, atau `gender` (+ neutral) |
| `--personas` | *(semua 40)* | Persona spesifik, pisah koma. Lebih diprioritaskan dari `--group` |
| `--samples_file` | `data/ethics/ETHICS_test_4subsets_10percent_from200.json` | File soal ETHICS |
| `--max_per_subset` | *(semua)* | Batasi jumlah soal per subset (mis. `10` untuk uji cepat) |
| `--skip_existing` | `False` | Lewati soal yang file JSON-nya sudah ada |
| `--max_tokens` | `4096` | Batas token output |
| `--sleep_sec` | `1.0` | Jeda antar request API (detik) |
| `--rebuild_csv_only` | `False` | Rebuild CSV dari JSON yang ada tanpa panggil API |
| `--parse_demo` | `False` | Tampilkan prompt + demo parsing, lalu keluar (tanpa API) |

> Uji cepat 1 model: tambahkan `--max_per_subset 5` agar hanya 5 soal/subset (20 soal total).

---

## Struktur Output

```
result/ETHICS/v9_pilot/Regional Cultural Persona/
└── {model_slug}/                       # e.g. deepseek_deepseek-r1-distill-qwen-32b
    ├── north_american_explicit/
    │   ├── ethics_commonsense_0001.json
    │   └── ...
    ├── north_american_implicit_1/
    │   ├── warmup/
    │   │   └── persona_warmup.json
    │   ├── ethics_commonsense_0001.json
    │   └── ...
    └── ... (16 persona)
```

CSV per-model otomatis di-rebuild ke:
`result/ETHICS/{model_slug}/result_ethics_v9_pilot_{model_slug}.csv`

---

## Setelah Selesai — Rebuild CSV Manual (opsional)

Jika perlu rebuild ulang tanpa run API:

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 31 --rebuild_csv_only
```

Untuk CSV per-grup (committed/abstain/missing):

```powershell
python src/rebuild/build_v9_csv_per_group.py --model deepseek_deepseek-r1-distill-qwen-32b
```
