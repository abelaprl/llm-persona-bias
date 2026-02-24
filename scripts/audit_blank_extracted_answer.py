import pandas as pd
import numpy as np

# Path to results.csv
csv_path = r"E:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think\results.csv"
df = pd.read_csv(csv_path, dtype=str)

def is_blank(val):
    if pd.isna(val):
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if not isinstance(val, str):
        return False
    s = val.strip().lower()
    return s == '' or s == 'none' or s == 'null'

# Apply blank check
blank_mask = df['extracted_answer'].apply(is_blank)
blank_rows = df[blank_mask]

total_rows = len(df)
total_blank = blank_mask.sum()

print(f"Total rows: {total_rows}")
print(f"Total blank extracted_answer: {total_blank}")

# Breakdown per persona_folder
breakdown = blank_rows['persona_folder'].value_counts()
print("\nBreakdown blank per persona_folder:")
for persona, count in breakdown.items():
    print(f"  {persona}: {count}")

# Top 10 examples
print("\nTop 10 contoh baris blank:")
cols = ['persona_folder', 'file', 'finish_reason', 'completion_tokens', 'response_content_full', 'error_category']
for idx, row in blank_rows.head(10).iterrows():
    content = row['response_content_full']
    if isinstance(content, str):
        content = content[:120].replace('\n', ' ')
    print(f"{row['persona_folder']}, {row['file']}, {row['finish_reason']}, {row['completion_tokens']}, {content}, {row['error_category']}")
