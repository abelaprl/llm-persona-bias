
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_dir', required=False, default=None, help='Directory containing results.csv and to write accuracy.csv')
    args = parser.parse_args()
    if args.report_dir:
        report_dir = args.report_dir
    else:
        report_dir = os.path.join(os.path.dirname(__file__), '../result/archive/reports_mixed_legacy/allenai_olmo-3-32b-think')
    results_path = os.path.join(report_dir, 'results.csv')
    persona_stats = {}
    total = 0
    correct = 0
    sum_prompt = 0
    sum_completion = 0
    sum_reasoning = 0
    sum_total = 0
    with open(results_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            persona = row['persona_folder']
            is_correct = row['is_correct'].strip().lower() == 'true'
            persona_stats.setdefault(persona, {'total': 0, 'correct': 0, 'prompt': 0, 'completion': 0, 'reasoning': 0, 'total_tokens': 0})
            persona_stats[persona]['total'] += 1
            if is_correct:
                persona_stats[persona]['correct'] += 1
            # Token stats (handle empty as 0)
            for k, col in [('prompt','prompt_tokens'),('completion','completion_tokens'),('reasoning','reasoning_tokens'),('total_tokens','total_tokens')]:
                try:
                    val = int(row[col]) if row[col] and row[col].isdigit() else 0
                except Exception:
                    val = 0
                persona_stats[persona][k] += val
            total += 1
            if is_correct:
                correct += 1
            sum_prompt += int(row['prompt_tokens']) if row['prompt_tokens'] and row['prompt_tokens'].isdigit() else 0
            sum_completion += int(row['completion_tokens']) if row['completion_tokens'] and row['completion_tokens'].isdigit() else 0
            sum_reasoning += int(row['reasoning_tokens']) if row['reasoning_tokens'] and row['reasoning_tokens'].isdigit() else 0
            sum_total += int(row['total_tokens']) if row['total_tokens'] and row['total_tokens'].isdigit() else 0
    # Write accuracy.csv
    out_path = os.path.join(report_dir, 'accuracy.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['persona_folder', 'correct', 'total', 'accuracy', 'sum_prompt_tokens', 'sum_completion_tokens', 'sum_reasoning_tokens', 'sum_total_tokens', 'avg_prompt_tokens', 'avg_completion_tokens', 'avg_reasoning_tokens', 'avg_total_tokens'])
        for persona, stat in persona_stats.items():
            acc = stat['correct'] / stat['total'] if stat['total'] else 0.0
            avg_prompt = stat['prompt'] / stat['total'] if stat['total'] else 0.0
            avg_completion = stat['completion'] / stat['total'] if stat['total'] else 0.0
            avg_reasoning = stat['reasoning'] / stat['total'] if stat['total'] else 0.0
            avg_total = stat['total_tokens'] / stat['total'] if stat['total'] else 0.0
            writer.writerow([persona, stat['correct'], stat['total'], f'{acc:.4f}', stat['prompt'], stat['completion'], stat['reasoning'], stat['total_tokens'], f'{avg_prompt:.2f}', f'{avg_completion:.2f}', f'{avg_reasoning:.2f}', f'{avg_total:.2f}'])
        # Write total
        acc = correct / total if total else 0.0
        avg_prompt = sum_prompt / total if total else 0.0
        avg_completion = sum_completion / total if total else 0.0
        avg_reasoning = sum_reasoning / total if total else 0.0
        avg_total = sum_total / total if total else 0.0
        writer.writerow(['ALL', correct, total, f'{acc:.4f}', sum_prompt, sum_completion, sum_reasoning, sum_total, f'{avg_prompt:.2f}', f'{avg_completion:.2f}', f'{avg_reasoning:.2f}', f'{avg_total:.2f}'])
    print(f'Wrote accuracy.csv to {out_path}')
    print(f'Total accuracy: {correct}/{total} = {acc:.4f}')
    print(f'Total prompt tokens: {sum_prompt}, avg: {avg_prompt:.2f}')
    print(f'Total completion tokens: {sum_completion}, avg: {avg_completion:.2f}')
    print(f'Total reasoning tokens: {sum_reasoning}, avg: {avg_reasoning:.2f}')
    print(f'Total tokens: {sum_total}, avg: {avg_total:.2f}')

if __name__ == '__main__':
    main()
