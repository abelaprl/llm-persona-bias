# Run Commands — ETHICS Prompt V7 / V8 / V9

Model: `deepseek/deepseek-r1-distill-qwen-32b` (order 43)  
Dataset: `data/ETHICS_test_4subsets_5each.json` (20 soal, 5 per subset)  
Persona: 25 total (neutral + 8 gender + 16 religion)

---

## Prompt V7

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ETHICS_test_4subsets_5each.json --skip_existing --categories neutral
```

**Gender** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ETHICS_test_4subsets_5each.json --skip_existing --categories gender_explicit,gender_implicit_1,gender_implicit_2,gender_implicit_3
```

**Religion** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_variants.py --model_order 43 --prompt_variants prompt_v7 --samples_file data/ETHICS_test_4subsets_5each.json --skip_existing --categories religion_explicit,religion_implicit_1,religion_implicit_2,religion_implicit_3
```

---

## Prompt V8

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas neutral
```

**Gender** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas man_explicit_1,man_implicit_1,man_implicit_2,man_implicit_3,woman_explicit_1,woman_implicit_1,woman_implicit_2,woman_implicit_3
```

**Religion** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_v8.py --model_order 43 --skip_existing --personas islam_explicit,islam_implicit_1,islam_implicit_2,islam_implicit_3,christian_explicit,christian_implicit_1,christian_implicit_2,christian_implicit_3,jewish_explicit,jewish_implicit_1,jewish_implicit_2,jewish_implicit_3,atheist_explicit,atheist_implicit_1,atheist_implicit_2,atheist_implicit_3
```

---

## Prompt V9

**Neutral**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --personas neutral
```

**Gender** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --group gender
```

**Religion** (explicit + implicit 1/2/3)
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --group religion
```
