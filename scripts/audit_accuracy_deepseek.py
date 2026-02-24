import pandas as pd
import os

base = r'E:/Tugas Akhir/TugasAkhir/result/reports/deepseek_deepseek-v3.2'
personas = [
    'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3',
    'woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3'
]

for p in personas:
    path = os.path.join(base, p, 'results.csv')
    df = pd.read_csv(path, dtype=str)
    n_true = (df['is_correct'].str.strip().str.lower() == 'true').sum()
    total = len(df)
    acc = n_true / total if total else 0.0
    print(f'{p}: {acc:.4f} ({n_true}/{total})')
