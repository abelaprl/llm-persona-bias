import csv
import os

# Path to the main results.csv
MAIN_CSV = r"e:\Tugas Akhir\TugasAkhir\result/archive/reports_mixed_legacy\deepseek_deepseek-v3.2\results.csv"
REPORTS_DIR = r"e:\Tugas Akhir\TugasAkhir\result/archive/reports_mixed_legacy\deepseek_deepseek-v3.2"

# Read all rows from the main CSV
with open(MAIN_CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Group rows by persona_folder
persona_rows = {}
for row in rows:
    persona = row[0]
    persona_rows.setdefault(persona, []).append(row)

# Write each persona's results.csv
for persona, rows in persona_rows.items():
    out_dir = os.path.join(REPORTS_DIR, persona)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
