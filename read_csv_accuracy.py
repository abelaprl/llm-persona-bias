import pandas as pd
from pathlib import Path

print("\n" + "="*140)
print("FINAL ACCURACY DARI CSV FILES (TRUTH SOURCE)")
print("="*140 + "\n")

csv_files = sorted(Path("results/reports").glob("*.csv"))

# Filter untuk yang tidak duplikat (yang dengan colon)
csv_files = [f for f in csv_files if not f.name.startswith("x-ai: Grok")]

results = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        if "Correct" in df.columns:
            correct = (df['Correct'] == 'Yes').sum()
            total = len(df)
            accuracy = correct / total * 100 if total > 0 else 0
            
            # Extract model and persona from filename
            name = csv_file.stem  # Remove .csv
            if "__" in name:
                model, persona = name.split("__")
            else:
                model, persona = name, "unknown"
            
            # Clean up model name
            model = model.replace("x-ai Grok 4.1 Fast", "Grok 4.1 Fast")
            model = model.replace("openrouter_bert-nebulon-alpha", "Nebulon Alpha")
            
            results.append({
                "Model": model,
                "Persona": persona,
                "Correct": correct,
                "Total": total,
                "Accuracy": accuracy
            })
            
            print(f"✓ {model:20} | {persona:20} | {correct:4}/{total:4} | {accuracy:6.2f}%")
    except Exception as e:
        print(f"❌ {csv_file.name}: {e}")

print(f"\n{'='*140}")
print("SUMMARY BY MODEL")
print(f"{'='*140}\n")

models_dict = {}
for r in results:
    model = r["Model"]
    if model not in models_dict:
        models_dict[model] = {"correct": 0, "total": 0}
    models_dict[model]["correct"] += r["Correct"]
    models_dict[model]["total"] += r["Total"]

for model, data in sorted(models_dict.items()):
    acc = data["correct"] / data["total"] * 100
    print(f"{model:20} | {data['correct']:5}/{data['total']:5} | {acc:6.2f}%")

# Overall
all_correct = sum(r["Correct"] for r in results)
all_total = sum(r["Total"] for r in results)
overall_acc = all_correct / all_total * 100

print(f"\n{'='*140}")
print(f"OVERALL: {all_correct}/{all_total} = {overall_acc:.2f}%")
print(f"{'='*140}\n")

