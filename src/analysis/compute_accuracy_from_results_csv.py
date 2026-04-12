import csv

CSV_PATH = r'result/reports/openai_gpt-oss-20b/results.csv'

def main():
    total = 0
    correct = 0
    with open(CSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('is_correct', '').strip().lower() == 'true':
                correct += 1
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.4f}")

if __name__ == '__main__':
    main()
