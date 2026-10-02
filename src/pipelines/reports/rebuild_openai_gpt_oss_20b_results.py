import os
import json
import csv
import sys
import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, repo_root)
from src.core.extract.rebuild_results_csv_root import extract_final_answer, get_response_text

RESULT_ROOT = os.path.join(repo_root, 'result', 'openai_gpt-oss-20b')
OUTPUT_CSV = os.path.join(repo_root, 'apply_extractor_openai_gpt_oss_20b.csv')

def iter_json_files(root):
    for subdir in os.listdir(root):
        subpath = os.path.join(root, subdir)
        if not os.path.isdir(subpath):
            continue
        for fname in os.listdir(subpath):
            if fname.endswith('.json'):
                yield subdir, fname, os.path.join(subpath, fname)

def main():
    rows = []
    for subdir, fname, fpath in iter_json_files(RESULT_ROOT):
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
            text = get_response_text(data.get('response', {}))
            answer, reason = extract_final_answer(text)
            rows.append({
                'subdir': subdir,
                'file': fname,
                'extracted_answer': answer,
                'reason': reason
            })
        except Exception as e:
            rows.append({
                'subdir': subdir,
                'file': fname,
                'extracted_answer': None,
                'reason': f'error: {e}'
            })
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subdir', 'file', 'extracted_answer', 'reason'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'Wrote {len(rows)} results to {OUTPUT_CSV}')

if __name__ == '__main__':
    main()
