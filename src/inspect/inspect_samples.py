"""Collect concrete representative samples for each error pattern."""
import pandas as pd, json, re, os, sys
sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

SAMPLES = [
    # (model_root, file_rel, gold, label)
    # total_keyword: "the total number ... is 100" but answer is 50 pairs
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00372.json', 50, '8b_total_100vs50'),
    # total_keyword: "scored a total of 18 points" but gold is 14
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00824.json', 14, '8b_total_18vs14'),
    # algebraic: "17 x 16 = 272" gold=272
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00155.json', 272, '8b_alg_272'),
    # algebraic: "300 / 10 = 30 shells. So, each girl brought 30" gold=300
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00535.json', 300, '8b_alg_300vs30'),
    # algebraic: "24 cents + 60 cents = 84 cents" gold=84
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00461.json', 84, '8b_alg_84'),
    # need_take: "last for 2 months" gold=1400
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00157.json', 1400, '8b_need_1400vs2'),
    # currency: "$6300 / $700 = 9" gold=9
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_explicit_1/gsm8k_00707.json', 9, '8b_cur_700vs9'),
    # currency: "$400 + $60 = $460" gold=460
    ('result/meta-llama_llama-3.1-8b-instruct', 'man_implicit_1/gsm8k_00010.json', 460, '8b_cur_460'),
    # 70b: "12 years ... profit in 13th year" gold=13 extracted=12
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00013.json', 13, '70b_13thyr'),
    # 70b: "store had sold 100 tennis shoes" gold=50 (pairs)
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00372.json', 50, '70b_total_100vs50'),
    # 70b algebraic: "10/2 = 5 loaves in the afternoon" gold=10 extracted=5
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00307.json', 10, '70b_alg_10vs5'),
    # 70b algebraic: "2x3=6 scoops" gold=6 extracted=2
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00712.json', 6, '70b_alg_6vs2'),
    # 70b need_take: "needs to order 1400 bananas to last for 2 months" extracted=2
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00157.json', 1400, '70b_need_1400vs2'),
    # 70b: diff is -25, gold=25
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_explicit_1/gsm8k_00781.json', 25, '70b_neg25'),
    # 70b: "has 398 toys, needs 2 more for 5 times" gold=398 extracted=5
    ('result/meta-llama_llama-3.3-70b-instruct', 'man_implicit_1/gsm8k_00550.json', 398, '70b_398vs5'),
]

for root, file_rel, gold, label in SAMPLES:
    json_path = os.path.join(root, file_rel)
    if not os.path.exists(json_path):
        print(f'[MISSING] {label}')
        continue
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    content = d.get('response', {}).get('content', '') or ''
    result, reason = extract_final_answer(content)
    status = 'OK' if result == gold else 'WRONG'
    print(f'[{status}] {label}: gold={gold} got={result} ({reason})')
    print(f'  TAIL (last 400):')
    print('  ' + content[-400:].replace('\n', '\n  '))
    print()
