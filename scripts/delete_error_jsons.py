import os
import json

# Directory containing the result files for woman_implicit_3
dir_path = r'e:\Tugas Akhir\TugasAkhir\result\google_gemma-3-27b-it\woman_implicit_3'

# Loop through all JSON files in the directory
for fname in os.listdir(dir_path):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(dir_path, fname)
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Check for error (e.g., HTTP 402 or success: false)
        if (
            ('response' in data and not data['response'].get('success', True)) or
            ('response' in data and data['response'].get('error', '').startswith('HTTP 402'))
        ):
            print(f"Deleting error file: {fname}")
            os.remove(fpath)
    except Exception as e:
        print(f"Skipping {fname}: {e}")
