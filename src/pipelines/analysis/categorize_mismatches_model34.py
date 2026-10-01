import json
from collections import defaultdict

def classify_issue(item):
    gt = item.get('gt')
    ex = item.get('extracted')
    detail = item.get('detail','') or ''
    tags = []
    if gt is None:
        tags.append('nan_gt')
    if ex is None:
        tags.append('no_extract')
    # detect boxed/latex in detail
    if '\\boxed' in detail or 'boxed' in detail or '\\[' in detail or '\\]' in detail:
        tags.append('latex_boxed')
    # detect currency or $ in detail
    if '$' in detail or 'dollars' in detail.lower() or '€' in detail or '£' in detail:
        tags.append('currency')
    # percent
    if '%' in detail or 'percent' in detail.lower():
        tags.append('percent')
    # negative
    if isinstance(gt, (int,float)) and gt < 0:
        tags.append('negative_gt')

    # numeric ratio analysis
    if isinstance(gt, (int,float)) and isinstance(ex, (int,float)) and gt != 0:
        ratio = ex / gt
        if abs(ratio - 1) < 0.01:
            tags.append('close')
        elif ratio == 0:
            tags.append('zero_extracted')
        elif ratio < 0.1:
            tags.append('small_fraction')
        elif ratio < 0.5:
            tags.append('less_than_half')
        elif ratio > 10:
            tags.append('much_larger')
        else:
            tags.append('other_ratio')

        # detect likely thousands/commas stripped
        if ex != 0:
            factor = gt / ex
            for f in [10,100,1000,10000]:
                if abs(factor - f) < 1e-6 or abs((factor/f) - 1) < 0.01:
                    tags.append(f'scale_factor_{f}')
    
    # detect intermediate tokens: small ints when GT large
    if isinstance(gt, (int,float)) and isinstance(ex, (int,float)) and gt >= 1000 and ex < 100:
        tags.append('picked_intermediate_small')

    if not tags:
        tags.append('unknown')
    return tags

def main():
    with open('analyze34_mismatches.json', 'r', encoding='utf-8') as f:
        items = json.load(f)

    counts = defaultdict(int)
    examples = defaultdict(list)

    for it in items:
        tags = classify_issue(it)
        for t in tags:
            counts[t] += 1
            if len(examples[t]) < 5:
                examples[t].append({'persona': it.get('persona'), 'file': it.get('file'), 'gt': it.get('gt'), 'ex': it.get('extracted'), 'detail': it.get('detail')[:200]})

    report = {'total_mismatches': len(items), 'counts': dict(counts), 'examples': examples}
    with open('mismatch_classification34.json', 'w', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)

    print('Wrote mismatch_classification34.json')
    print('\nSummary counts:')
    for k,v in sorted(counts.items(), key=lambda x:-x[1]):
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
