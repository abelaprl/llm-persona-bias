import csv
import os

def main():
    # Path to results.csv
    results_path = os.path.join(os.path.dirname(__file__), '../result/reports/allenai_olmo-3-32b-think/results.csv')
    persona_stats = {}
    total = 0
    correct = 0
    with open(results_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            persona = row['persona_folder']
            is_correct = row['is_correct'].strip().lower() == 'true'
            persona_stats.setdefault(persona, {'total': 0, 'correct': 0})
            persona_stats[persona]['total'] += 1
            if is_correct:
                persona_stats[persona]['correct'] += 1
            total += 1
            if is_correct:
                correct += 1
    # Write accuracy.csv
    out_path = os.path.join(os.path.dirname(__file__), '../result/reports/allenai_olmo-3-32b-think/accuracy.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['persona_folder', 'correct', 'total', 'accuracy'])
        for persona, stat in persona_stats.items():
            acc = stat['correct'] / stat['total'] if stat['total'] else 0.0
            writer.writerow([persona, stat['correct'], stat['total'], f'{acc:.4f}'])
        # Write total
        acc = correct / total if total else 0.0
        writer.writerow(['ALL', correct, total, f'{acc:.4f}'])
    print(f'Wrote accuracy.csv to {out_path}')

if __name__ == '__main__':
    main()
