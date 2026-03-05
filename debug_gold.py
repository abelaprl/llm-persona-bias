import json, re
for qid, fn in [('00189', 'man_explicit_1/gsm8k_00189'), ('00289', 'man_explicit_1/gsm8k_00289'), ('00158', 'man_explicit_1/gsm8k_00158')]:
    with open(f'result/meta-llama_llama-3.3-70b-instruct/{fn}.json', encoding='utf-8') as f:
        d = json.load(f)
    gt = d['input']['ground_truth']
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', gt)
    gold = m.group(1) if m else '?'
    print(f'{fn}: gold={gold}')
    
    # Also check last 5 lines of response
    content = d['response']['content']
    lines = content.splitlines()
    print('  last 5 lines:')
    for l in lines[-5:]:
        print(f'    {repr(l)}')
    print()
