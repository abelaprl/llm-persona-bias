"""
Jalankan semua model order 31-49 untuk 1 soal GSM8K dengan 1 persona
Sequential execution untuk monitoring progress
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

def load_models():
    """Load model configuration order 31-49"""
    with open('config/model_keys.json', 'r', encoding='utf-8') as f:
        models = json.load(f)
    return [m for m in models if 31 <= m['order'] <= 49]

def run_single_model(model_order, gsm8k_id, persona_id):
    """Run experiment for single model"""
    cmd = [
        'python',
        'src/gsm8k_scenario_a.py',
        '--model_order', str(model_order),
        '--gsm8k_id', str(gsm8k_id),
        '--persona_id', str(persona_id)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'model_order': model_order,
        'exit_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def main():
    gsm8k_id = 1
    persona_id = 1
    
    print("="*80)
    print(f"BATCH EXECUTION: All Models (31-49)")
    print(f"GSM8K ID: {gsm8k_id} | Persona ID: {persona_id}")
    print("="*80)
    
    # Load models
    models = load_models()
    print(f"\nFound {len(models)} models to test\n")
    
    results = []
    success_count = 0
    failed_count = 0
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] Model Order {model['order']}: {model['name']}")
        print(f"{'='*80}")
        
        try:
            result = run_single_model(model['order'], gsm8k_id, persona_id)
            
            if result['exit_code'] == 0:
                print(f"✓ SUCCESS")
                success_count += 1
                results.append({
                    'model_order': model['order'],
                    'model_name': model['name'],
                    'status': 'SUCCESS'
                })
            else:
                print(f"✗ FAILED (exit code: {result['exit_code']})")
                print(f"Error: {result['stderr'][:200]}")
                failed_count += 1
                results.append({
                    'model_order': model['order'],
                    'model_name': model['name'],
                    'status': 'FAILED',
                    'error': result['stderr'][:200]
                })
        
        except Exception as e:
            print(f"✗ EXCEPTION: {str(e)}")
            failed_count += 1
            results.append({
                'model_order': model['order'],
                'model_name': model['name'],
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"\n✓ SUCCESS: {success_count}/{len(models)} models")
    print(f"✗ FAILED:  {failed_count}/{len(models)} models")
    
    if failed_count > 0:
        print(f"\nFailed models:")
        for r in results:
            if r['status'] in ['FAILED', 'ERROR']:
                print(f"  [{r['model_order']}] {r['model_name']}: {r.get('error', 'Unknown error')[:100]}")
    
    # Save summary
    summary_file = f"logs_batch_gsm8k_id{gsm8k_id}_persona{persona_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"\n{'='*80}")
    print(f"BATCH EXECUTION COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
