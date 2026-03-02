# Accuracy Analysis Report

## FINAL RESULTS (dari CSV files - Truth Source)

### Per Model + Persona
| Model | Persona | Correct | Total | Accuracy |
|-------|---------|---------|-------|----------|
| Grok 4.1 Fast | man_implicits | 446 | 473 | 94.29% |
| Grok 4.1 Fast | woman_implicits | 421 | 450 | 93.56% |
| Nebulon Alpha | man_implicits | 375 | 394 | 95.18% |
| Nebulon Alpha | woman_implicits | 346 | 363 | 95.32% |

### Summary by Model
| Model | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Grok 4.1 Fast | 867 | 923 | 93.93% |
| Nebulon Alpha | 721 | 757 | 95.24% |
| **OVERALL** | **1588** | **1680** | **94.52%** |

## Files dalam direktori ini

### CSV Files (Detailed Results)
- `x-ai Grok 4.1 Fast__man_implicits.csv` - Grok man_implicits dengan 473 soal
- `x-ai Grok 4.1 Fast__woman_implicits.csv` - Grok woman_implicits dengan 450 soal
- `openrouter_bert-nebulon-alpha__man_implicits.csv` - Nebulon man_implicits dengan 394 soal
- `openrouter_bert-nebulon-alpha__woman_implicits.csv` - Nebulon woman_implicits dengan 363 soal

### Excel Files (Formatted)
- `x-ai Grok 4.1 Fast__man_implicits.xlsx`
- `x-ai Grok 4.1 Fast__woman_implicits.xlsx`
- `openrouter_bert-nebulon-alpha__man_implicits.xlsx`
- `openrouter_bert-nebulon-alpha__woman_implicits.xlsx`

## Kolom dalam CSV

| Kolom | Deskripsi |
|-------|-----------|
| Index | Question ID dari GSM8K dataset |
| Question | Teks soal |
| Expected | Jawaban yang benar |
| Model Answer | Jawaban dari model |
| Correct | "Yes" atau "No" |

## Cara Membaca CSV

### Opsi 1: Buka dengan Excel/LibreOffice
Langsung double-click file .xlsx atau .csv

### Opsi 2: Terminal
```bash
head -20 "x-ai Grok 4.1 Fast__man_implicits.csv"
```

### Opsi 3: Python
```python
import pandas as pd
df = pd.read_csv("x-ai Grok 4.1 Fast__man_implicits.csv")
print(df.head())
print((df['Correct'] == 'Yes').sum() / len(df) * 100)
```

## Mengapa Persentase Berubah-ubah?

1. **Jumlah file terus bertambah** - Eksperimen masih berjalan di background
2. **Index berbeda per model** - Nemotron dari 0, Grok/Nebulon dari 300
3. **Multiple script versions** - Ada beberapa versi script analisis dengan logic berbeda

⚠️ **CSV files adalah TRUTH SOURCE** - Angka di CSV tidak akan berubah karena sudah di-save.

Akurasi baru hanya muncul jika kita re-run analisis dengan file yang lebih banyak.

## Data Quality Notes

- GSM8K test set: ~7473 questions total
- Subset used: Index 300-1599 (~1300 questions)
- Grok tested: Q300-Q1087 range
- Nebulon tested: Q300-Q721 range
- Answer format: "Final answer: X" dalam response model
- Tolerance: 0.001 (untuk floating point comparison)
