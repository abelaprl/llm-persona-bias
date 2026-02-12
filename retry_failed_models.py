"""
Retry failed models: order 31 and 34
"""
import subprocess

failed_models = [31, 34]
gsm8k_id = 1
persona_id = 1

print("="*80)
print(f"RETRY FAILED MODELS: {failed_models}")
print(f"GSM8K ID: {gsm8k_id} | Persona ID: {persona_id}")
print("="*80)

for model_order in failed_models:
    print(f"\n[Retrying] Model Order {model_order}...")
    
    cmd = [
        'python',
        'src/gsm8k_scenario_a.py',
        '--model_order', str(model_order),
        '--gsm8k_id', str(gsm8k_id),
        '--persona_id', str(persona_id)
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"[OK] Model {model_order} SUCCESS")
    else:
        print(f"[FAILED] Model {model_order} FAILED")

print(f"\n{'='*80}")
print("RETRY COMPLETED")
print(f"{'='*80}")
