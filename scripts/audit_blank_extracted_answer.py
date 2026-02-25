import pandas as pd
import numpy as np
import os

RESULTS_PATH = r"E:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think\results.csv"
CSV_BLANK_PATH = r"E:\Tugas Akhir\TugasAkhir\result\reports\allenai_olmo-3-32b-think\csv_blank_extracted.csv"

# Read CSV
# Use dtype=str to preserve all as string, na_values for null/None
na_vals = ["", " ", "None", "null", None, np.nan]
df = pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)

# Define blank criteria for extracted_answer
blank_mask = df["extracted_answer"].apply(lambda x: (pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() in ["none", "null"]))

# Total rows
total_rows = len(df)
# Total blank
total_blank = blank_mask.sum()

# Breakdown per persona_folder
breakdown = df.loc[blank_mask, "persona_folder"].value_counts().sort_index()

# Top 10 blank examples
top10 = df.loc[blank_mask].head(10).copy()
# Truncate response_content_full
if "response_content_full" in top10.columns:
    top10["response_content_full"] = top10["response_content_full"].apply(lambda x: str(x)[:120] if pd.notna(x) else "")

# Print audit summary
print(f"Total rows: {total_rows}")
print(f"Total blank extracted_answer: {total_blank}")
print("Breakdown blank per persona_folder:")
for persona, count in breakdown.items():
    print(f"  {persona}: {count}")
print("\nTop 10 blank examples:")
print(top10[["persona_folder", "file", "finish_reason", "completion_tokens", "response_content_full", "error_category"]].to_string(index=False))

# Save all blank rows to csv_blank_extracted.csv
blank_df = df.loc[blank_mask].copy()
blank_df.to_csv(CSV_BLANK_PATH, index=False)
print(f"\nWrote blank rows to {CSV_BLANK_PATH}")
