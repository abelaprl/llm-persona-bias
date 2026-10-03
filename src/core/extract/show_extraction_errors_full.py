import csv

CSV_PATH = r'result/archive/reports_mixed_legacy/openai_gpt-oss-20b/results.csv'

# Print a few examples where extraction failed (is_correct == False), show full answer and analysis

def main():
    count = 0
    max_show = 100
    with open(CSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('is_correct', '').strip().lower() == 'false':
                print(f"File: {row['file']}")
                print(f"Persona: {row['persona_folder']}")
                print(f"GT: {row['ground_truth_answer']} | Extracted: {row['extracted_answer']}")
                print(f"Extraction reason: {row['extraction_reason']}")
                print(f"Error category: {row['error_category']}")
                print("FULL ANSWER:")
                print(row['response_content_full'])
                print('-'*60)
                count += 1
                if count >= max_show:
                    break
    print(f"Total incorrect shown: {count}")

if __name__ == '__main__':
    main()
