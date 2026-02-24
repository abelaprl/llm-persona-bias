import os
import csv

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_folders = [
        "man_explicit_1",
        "man_implicit_1",
        "man_implicit_2",
        "man_implicit_3",
        "woman_explicit_1",
        "woman_implicit_1",
        "woman_implicit_2",
        "woman_implicit_3",
    ]
    output_file = os.path.join(base_dir, "results_full.csv")
    header = None
    total_rows = 0
    with open(output_file, "w", newline='', encoding="utf-8") as fout:
        writer = None
        for idx, persona in enumerate(persona_folders):
            persona_file = os.path.join(base_dir, persona, "results.csv")
            if not os.path.exists(persona_file):
                print(f"Warning: {persona_file} does not exist, skipping.")
                continue
            with open(persona_file, "r", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                file_header = next(reader)
                if header is None:
                    header = file_header
                    writer = csv.writer(fout)
                    writer.writerow(header)
                else:
                    # Check header consistency
                    if file_header != header:
                        print(f"Header mismatch in {persona_file}")
                        continue
                row_count = 0
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
                print(f"Merged {row_count} rows from {persona}")
                total_rows += row_count
    print(f"Total rows merged: {total_rows}")

if __name__ == "__main__":
    main()