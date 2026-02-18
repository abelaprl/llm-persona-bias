import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extractor34
p = Path('result/openai_gpt-oss-120b/man_explicit_1/gsm8k_00344.json')
data = json.loads(p.read_text(encoding='utf-8'))
text = data.get('response', {}).get('content')
print('RESPONSE:\n', text)
val = extractor34.extract_final_answer(text)
print('\nEXTRACTED:', val)
print('\nGROUND TRUTH from input.ground_truth:\n', data.get('input', {}).get('ground_truth'))
