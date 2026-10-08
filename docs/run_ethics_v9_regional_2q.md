# Run Commands — ETHICS V9 Regional Persona (2 soal/subset)

Model: `deepseek/deepseek-r1-distill-qwen-32b` (order 43)  
Dataset: `data/ethics/ETHICS_test_4subsets_10percent_from200.json`

Setelah reorganisasi `data/`, tambahkan `--samples_file data/ethics/ETHICS_test_4subsets_10percent_from200.json` pada setiap perintah runner di bawah. Perubahan default runner menunggu izin sesuai AGENTS.md.
Limit: `--max_per_subset 2` → 2 soal × 4 subset = **8 soal per persona**  
Persona regional: 17 total (neutral + 4×4 regional)

---

## Dry run (cek prompt + parsing, tanpa API call)

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --parse_demo
```

---

## Neutral

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas neutral
```

---

## North American

**Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas north_american_explicit
```

**Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas north_american_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas north_american_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas north_american_implicit_3
```

---

## East Asian

**Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas east_asian_explicit
```

**Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas east_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas east_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas east_asian_implicit_3
```

---

## South Asian

**Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas south_asian_explicit
```

**Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas south_asian_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas south_asian_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas south_asian_implicit_3
```

---

## Sub-Saharan African

**Explicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas sub_saharan_african_explicit
```

**Implicit**
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas sub_saharan_african_implicit_1
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas sub_saharan_african_implicit_2
```
```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --personas sub_saharan_african_implicit_3
```

---

## Semua regional sekaligus (1 command)

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --skip_existing --max_per_subset 2 --group regional
```

---

## Rebuild CSV saja (tanpa API call)

```powershell
python src/ethics_runner/run_ethics_prompt_v9.py --model_order 43 --rebuild_csv_only
```
