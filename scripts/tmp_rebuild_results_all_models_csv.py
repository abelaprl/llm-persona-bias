import csv
import shutil
from datetime import datetime
from pathlib import Path

reports = Path(r"E:\Tugas Akhir\TugasAkhir\result\reports")
out_file = reports / "results_all_models_all_personas.csv"

models = [
    "allenai_olmo-3-32b-think",
    "allenai_olmo-3.1-32b-instruct",
    "deepseek_deepseek-v3.2",
    "google_gemini-2.5-flash",
    "google_gemma-2-9b-it",
    "google_gemma-3-27b-it",
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "openai_gpt-oss-120b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-235b-a22b-2507",
    "qwen_qwen3-next-80b-a3b-thinking",
]

sources = []
for model in models:
    src = reports / model / f"result_{model.replace('.', '_')}.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing source for {model}: {src}")
    sources.append((model, src))

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if out_file.exists():
    bad_backup = reports / f"results_all_models_all_personas.before_csvfix_{stamp}.csv"
    shutil.copy2(out_file, bad_backup)

# Use the first source as schema reference so columns match canonical per-model CSV shape.
with open(sources[0][1], "r", encoding="utf-8-sig", newline="") as f0:
    first_reader = csv.DictReader(f0)
    source_fields = list(first_reader.fieldnames or [])

fieldnames = ["model_name"] + source_fields

total_rows = 0
stats = []

with open(out_file, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for model, src in sources:
        rows = 0
        with open(src, "r", encoding="utf-8-sig", newline="") as fin:
            reader = csv.DictReader(fin)
            in_fields = list(reader.fieldnames or [])
            if in_fields != source_fields:
                raise ValueError(
                    f"Schema mismatch in {src.name}.\n"
                    f"Expected: {source_fields}\n"
                    f"Actual:   {in_fields}"
                )

            for rec in reader:
                out = {k: rec.get(k, "") for k in source_fields}
                out["model_name"] = model
                writer.writerow(out)
                rows += 1

        stats.append((model, rows, src.name))
        total_rows += rows

print(f"Wrote: {out_file}")
print(f"Models merged: {len(sources)}")
print(f"Total rows: {total_rows}")
for model, rows, name in stats:
    print(f"{model}\t{rows}\t{name}")
