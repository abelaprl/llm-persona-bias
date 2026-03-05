import json
import sys
sys.path.insert(0, 'src')
from rebuild_results_csv_gemini import extract_final_answer

# Test on a working case first
path1 = r'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\man_explicit_1\gsm8k_00020.json'
with open(path1) as f:
    data = json.load(f)
    content = data.get('response', {}).get('content', '')
    result = extract_final_answer(content)
    print(f'Working case (gsm8k_00020):')
    print(f'  Response length: {len(content)}')
    print(f'  Extracted: {repr(result)}')
    print(f'  Last 300 chars: {content[-300:]}')

# Now test on a no_extract case
path2 = r'E:\Tugas Akhir\TugasAkhir\result\google_gemma-2-9b-it\man_explicit_1\gsm8k_00154.json'
with open(path2) as f:
    data = json.load(f)
    content = data.get('response', {}).get('content', '')
    result = extract_final_answer(content)
    print(f'\n\nNo_extract case (gsm8k_00154):')
    print(f'  Response length: {len(content)}')
    print(f'  Extracted: {repr(result)}')
    print(f'  Last 300 chars: {content[-300:]}')
