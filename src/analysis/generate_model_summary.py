"""
Generate summary CSV for models that have completed all 8 personas
Output: result/reports/summary_{model_name}.csv
"""
import csv
from pathlib import Path

EXPECTED_PERSONAS = [
    'man_explicit_1',
    'man_implicit_1', 'man_implicit_2', 'man_implicit_3',
    'woman_explicit_1', 
    'woman_implicit_1', 'woman_implicit_2', 'woman_implicit_3'
]

def calculate_persona_stats(csv_path):
    """Calculate stats from a persona CSV file"""
    total = 0
    correct = 0
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('is_correct', '').lower() == 'true':
                correct += 1
            
            # Sum tokens
            try:
                prompt_tokens += int(float(row.get('prompt_tokens', 0) or 0))
            except:
                pass
            try:
                completion_tokens += int(float(row.get('completion_tokens', 0) or 0))
            except:
                pass
            try:
                reasoning_tokens += int(float(row.get('reasoning_tokens', 0) or 0))
            except:
                pass
            try:
                total_tokens += int(float(row.get('total_tokens', 0) or 0))
            except:
                pass
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return {
        'total_questions': total,
        'correct': correct,
        'accuracy': round(accuracy, 2),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens
    }


def check_model_complete(model_dir):
    """Check if model has all 8 persona CSVs"""
    for persona in EXPECTED_PERSONAS:
        csv_path = model_dir / persona / "results.csv"
        if not csv_path.exists():
            return False
    return True


def generate_model_summary(model_dir, output_path):
    """Generate summary CSV for a model"""
    model_name = model_dir.name
    
    rows = []
    
    for persona in EXPECTED_PERSONAS:
        csv_path = model_dir / persona / "results.csv"
        if csv_path.exists():
            stats = calculate_persona_stats(csv_path)
            rows.append({
                'model': model_name,
                'persona': persona,
                'total_questions': stats['total_questions'],
                'correct': stats['correct'],
                'accuracy_pct': stats['accuracy'],
                'prompt_tokens': stats['prompt_tokens'],
                'completion_tokens': stats['completion_tokens'],
                'reasoning_tokens': stats['reasoning_tokens'],
                'total_tokens': stats['total_tokens']
            })
    
    # Write summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['model', 'persona', 'total_questions', 'correct', 'accuracy_pct',
                  'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def main():
    reports_dir = Path("result/reports")
    
    print("="*80, flush=True)
    print("GENERATING MODEL SUMMARIES", flush=True)
    print("="*80, flush=True)
    print(flush=True)
    
    # Get all model folders
    model_folders = [f for f in reports_dir.iterdir() if f.is_dir()]
    
    complete_models = []
    incomplete_models = []
    
    for model_dir in sorted(model_folders):
        if check_model_complete(model_dir):
            complete_models.append(model_dir)
        else:
            # Count how many CSVs exist
            count = sum(1 for p in EXPECTED_PERSONAS if (model_dir / p / "results.csv").exists())
            incomplete_models.append((model_dir.name, count))
    
    print(f"Models with 8 CSVs: {len(complete_models)}", flush=True)
    print(f"Incomplete models: {len(incomplete_models)}", flush=True)
    print(flush=True)
    
    if incomplete_models:
        print("Incomplete:", flush=True)
        for name, count in incomplete_models:
            print(f"  {name}: {count}/8 CSVs", flush=True)
        print(flush=True)
    
    # Generate summaries for complete models
    for model_dir in complete_models:
        output_path = reports_dir / f"summary_{model_dir.name}.csv"
        count = generate_model_summary(model_dir, output_path)
        print(f"[OK] {model_dir.name}: {count} personas -> {output_path}", flush=True)
    
    print(flush=True)
    print("="*80, flush=True)
    print(f"DONE! Generated {len(complete_models)} summary files", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
