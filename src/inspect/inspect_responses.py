import json

cases = [
    'man_explicit_1/gsm8k_00154.json',
    'man_explicit_1/gsm8k_00162.json',
    'man_explicit_1/gsm8k_00173.json',
]

for case in cases:
    path = rf'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\{case}'
    try:
        with open(path) as f:
            data = json.load(f)
            content = data.get('response', {}).get('content', '')
            print(f'\n{"="*80}')
            print(f'File: {case}')
            print(f'{"="*80}')
            print('Last 800 chars of response:')
            print(content[-800:])
    except Exception as e:
        print(f'{case}: ERROR - {e}')
