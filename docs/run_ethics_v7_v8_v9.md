# Run Commands — ETHICS Prompt V7 / V8 / V9

Model: `deepseek/deepseek-r1-distill-qwen-32b` (order 43)  
Dataset V7: `data/ethics/ETHICS_test_4subsets_5each.json` (20 soal, 5 per subset) — di-pass eksplisit via `--samples_file`
Dataset V9: `data/ethics/ETHICS_test_4subsets_10percent_from200.json` — tambahkan `--samples_file data/ethics/ETHICS_test_4subsets_10percent_from200.json` pada setiap perintah v9 di bawah; perubahan default runner menunggu izin sesuai AGENTS.md.
Persona: 41 total (neutral + 8 gender + 16 religion + 16 regional)

---

## Prompt V7

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories neutral
```

**Gender Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_explicit --personas man_explicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_explicit --personas woman_explicit_1
```

**Gender Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_1 --personas man_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_2 --personas man_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_3 --personas man_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_1 --personas woman_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_2 --personas woman_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_implicit_3 --personas woman_implicit_3
```

**Religion Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_explicit --personas islam_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_explicit --personas christian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_explicit --personas jewish_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_explicit --personas atheist_explicit
```

**Religion Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_1 --personas islam_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_1 --personas christian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_1 --personas jewish_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_1 --personas atheist_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_2 --personas islam_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_2 --personas christian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_2 --personas jewish_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_2 --personas atheist_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_3 --personas islam_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_3 --personas christian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_3 --personas jewish_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_implicit_3 --personas atheist_implicit_3
```

**Regional Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_explicit --personas western_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_explicit --personas east_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_explicit --personas south_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_explicit --personas african_explicit
```

**Regional Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_1 --personas western_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_1 --personas east_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_1 --personas south_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_1 --personas african_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_2 --personas western_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_2 --personas east_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_2 --personas south_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_2 --personas african_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_3 --personas western_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_3 --personas east_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_3 --personas south_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ethics/ETHICS_test_4subsets_5each.json --skip_existing --categories regional_implicit_3 --personas african_implicit_3
```

---

## Prompt V8

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas neutral
```

**Gender Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas man_explicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas woman_explicit_1
```

**Gender Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas man_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas man_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas man_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas woman_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas woman_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas woman_implicit_3
```

**Religion Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas islam_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas christian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas jewish_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas atheist_explicit
```

**Religion Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas islam_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas christian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas jewish_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas atheist_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas islam_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas christian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas jewish_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas atheist_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas islam_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas christian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas jewish_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas atheist_implicit_3
```

**Regional Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas western_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas east_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas south_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas african_explicit
```

**Regional Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas western_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas east_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas south_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas african_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas western_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas east_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas south_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas african_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas western_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas east_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas south_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas african_implicit_3
```

---

## Prompt V9

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas neutral
```

**Gender Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas man_explicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas woman_explicit_1
```

**Gender Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas man_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas man_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas man_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas woman_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas woman_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas woman_implicit_3
```

**Religion Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas islam_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas christian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas jewish_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas atheist_explicit
```

**Religion Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas islam_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas christian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas jewish_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas atheist_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas islam_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas christian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas jewish_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas atheist_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas islam_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas christian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas jewish_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas atheist_implicit_3
```

**Regional Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas western_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas east_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas south_asian_explicit
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas african_explicit
```

**Regional Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas western_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas east_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas south_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas african_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas western_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas east_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas south_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas african_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas western_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas east_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas south_asian_implicit_3
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas african_implicit_3
```
