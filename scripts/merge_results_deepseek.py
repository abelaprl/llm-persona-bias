import pandas as pd
import os

base = r'E:/Tugas Akhir/TugasAkhir/result/reports/deepseek_deepseek-v3.2'
personas = [
    'man_explicit_1','man_implicit_1','man_implicit_2','man_implicit_3',
    'woman_explicit_1','woman_implicit_1','woman_implicit_2','woman_implicit_3'
]

all_dfs = []
for p in personas:
    path = os.path.join(base, p, 'results.csv')
    df = pd.read_csv(path, dtype=str)
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_path = os.path.join(base, 'results_full.csv')
full_df.to_csv(full_path, index=False)
print(f'Wrote merged results to {full_path} (total rows: {len(full_df)})')
