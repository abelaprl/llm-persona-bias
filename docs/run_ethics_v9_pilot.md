# Cara Menjalankan — ETHICS v9_pilot per Persona

Runner: `src/ethics_runner/run_ethics_prompt_v9.py`
Output: `result/ETHICS/v9_pilot/{Group Persona}/{model_slug}/{persona}/`

> Jalankan dari folder `D:\TugasAkhir`.
> Copy **satu blok**, Enter, tunggu `Done.`, baru lanjut ke blok berikutnya.
> Ganti `--model_order 43` dengan nomor model yang diinginkan.
> Setiap command memuat **1 persona** saja (commonsense 388 + deontology 359 + justice 270 + virtue 497 = **1514 soal**), bukan satu grup penuh.

---

## Daftar Model

| order | Nama Model |
|------:|-----------|
| 31 | OpenAI: GPT-OSS-20B |
| 32 | Qwen: Qwen3 Next 80B A3B (Thinking) |
| 33 | Qwen: Qwen3 235B A22B Instruct 2507 |
| 34 | OpenAI: GPT-OSS-120B |
| 35 | Meta: Llama 3.3 70B Instruct |
| 36 | Google: Gemma 3 27B |
| 37 | Google: Gemma 2 9B (Groq) |
| 38 | DeepSeek V3.2 |
| 39 | Google: Gemini 2.5 Flash |
| 40 | Meta: Llama 3.1 8B Instruct |
| 41 | Mistral: Ministral 8B |
| 42 | Allen AI: OLMo 3.1 32B Instruct |
| **43** | **DeepSeek: R1 Distill Qwen 32B** |
| 44 | Arcee AI: Trinity Large Thinking |
| 45 | MoonshotAI: Kimi K2 Thinking |
| 46 | NVIDIA: Nemotron Nano 9B V2 |
| 47 | Qwen: Qwen3 8B |

---

## Neutral Persona

**neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas neutral --skip_existing
```

---

## Gender Persona

**man_explicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas man_explicit_1 --skip_existing
```

**man_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas man_implicit_1 --skip_existing
```

**man_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas man_implicit_2 --skip_existing
```

**man_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas man_implicit_3 --skip_existing
```

**woman_explicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas woman_explicit_1 --skip_existing
```

**woman_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas woman_implicit_1 --skip_existing
```

**woman_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas woman_implicit_2 --skip_existing
```

**woman_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas woman_implicit_3 --skip_existing
```

---

## Religion Persona

**islam_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas islam_explicit --skip_existing
```

**islam_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas islam_implicit_1 --skip_existing
```

**islam_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas islam_implicit_2 --skip_existing
```

**islam_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas islam_implicit_3 --skip_existing
```

**christian_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas christian_explicit --skip_existing
```

**christian_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas christian_implicit_1 --skip_existing
```

**christian_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas christian_implicit_2 --skip_existing
```

**christian_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas christian_implicit_3 --skip_existing
```

**jewish_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas jewish_explicit --skip_existing
```

**jewish_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas jewish_implicit_1 --skip_existing
```

**jewish_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas jewish_implicit_2 --skip_existing
```

**jewish_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas jewish_implicit_3 --skip_existing
```

**atheist_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas atheist_explicit --skip_existing
```

**atheist_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas atheist_implicit_1 --skip_existing
```

**atheist_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas atheist_implicit_2 --skip_existing
```

**atheist_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas atheist_implicit_3 --skip_existing
```

---

## Regional Persona

**north_american_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas north_american_explicit --skip_existing
```

**north_american_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas north_american_implicit_1 --skip_existing
```

**north_american_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas north_american_implicit_2 --skip_existing
```

**north_american_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas north_american_implicit_3 --skip_existing
```

**east_asian_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas east_asian_explicit --skip_existing
```

**east_asian_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas east_asian_implicit_1 --skip_existing
```

**east_asian_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas east_asian_implicit_2 --skip_existing
```

**east_asian_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas east_asian_implicit_3 --skip_existing
```

**south_asian_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas south_asian_explicit --skip_existing
```

**south_asian_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas south_asian_implicit_1 --skip_existing
```

**south_asian_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas south_asian_implicit_2 --skip_existing
```

**south_asian_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas south_asian_implicit_3 --skip_existing
```

**sub_saharan_african_explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas sub_saharan_african_explicit --skip_existing
```

**sub_saharan_african_implicit_1**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas sub_saharan_african_implicit_1 --skip_existing
```

**sub_saharan_african_implicit_2**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas sub_saharan_african_implicit_2 --skip_existing
```

**sub_saharan_african_implicit_3**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --personas sub_saharan_african_implicit_3 --skip_existing
```

---

## Rebuild CSV (setelah semua persona selesai)

> Cukup sekali per model — scan semua JSON di `v9_pilot/<model>/`, tidak per grup.

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --rebuild_csv_only
```
