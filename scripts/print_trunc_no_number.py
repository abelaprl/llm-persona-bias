import csv
import os

def main():
    results_path = os.path.join(os.path.dirname(__file__), '../result/reports/allenai_olmo-3-32b-think/results.csv')
    with open(results_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['error_category'] in ('truncated_output', 'no_number_found'):
                print(f"{row['persona_folder']},{row['file']},{row['error_category']}")

if __name__ == '__main__':
    main()
