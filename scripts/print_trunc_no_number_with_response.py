import csv
import os
import json

def main():
    results_path = os.path.join(os.path.dirname(__file__), '../result/reports/allenai_olmo-3-32b-think/results.csv')
    input_root = os.path.join(os.path.dirname(__file__), '../result/allenai_olmo-3-32b-think')
    with open(results_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['error_category'] in ('truncated_output', 'no_number_found'):
                persona = row['persona_folder']
                file_rel = row['file'].replace('\\', os.sep).replace('/', os.sep)
                json_path = os.path.join(input_root, persona, file_rel)
                try:
                    with open(json_path, encoding='utf-8') as jf:
                        data = json.load(jf)
                    resp = data.get('response', {})
                    print(f"{persona},{file_rel},{row['error_category']}")
                    print("--- RESPONSE ---")
                    print(json.dumps(resp, ensure_ascii=False, indent=2)[:2000])
                    print("---------------\n")
                except Exception as e:
                    print(f"{persona},{file_rel},{row['error_category']} (FAILED TO LOAD: {e})\n")

if __name__ == '__main__':
    main()
