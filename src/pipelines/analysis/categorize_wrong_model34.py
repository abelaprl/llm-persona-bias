import re, collections

IN_PATH = 'wrong_examples_model34.txt'

def parse_line(line):
    # format: persona/filename | GT: <gt> | Extracted: <ex> | Snippet: <snip>
    parts = line.split('|')
    if len(parts) < 4:
        return None
    filepart = parts[0].strip()
    gt_part = parts[1].strip()
    ex_part = parts[2].strip()
    snip = '|'.join(parts[3:]).strip()
    try:
        gt = float(re.search(r'GT:\s*([0-9.,+-]+)', gt_part).group(1).replace(',',''))
    except:
        gt = None
    try:
        ex = float(re.search(r'Extracted:\s*([0-9.,+-]+)', ex_part).group(1).replace(',',''))
    except:
        ex = None
    return filepart, gt, ex, snip


def categorize(gt, ex, snip):
    if ex is None:
        return 'no_extract'
    if gt is None:
        return 'no_gt'
    if ex == 0:
        if '\\boxed' in snip or 'boxed' in snip or '\\[' in snip or '\\]' in snip:
            return 'missed_boxed_zero'
        return 'extracted_zero'
    # relative error
    if gt != 0:
        ratio = ex / gt
    else:
        ratio = float('inf') if ex!=0 else 1.0
    # heuristics for categories
    if 0.01 <= ratio <= 0.1:
        return 'much_smaller'
    if 0.1 < ratio < 0.5:
        return 'smaller'
    if 0.5 <= ratio <= 0.9:
        return 'slightly_smaller'
    if 0.9 < ratio < 1.1:
        return 'correct-ish_but_counted_wrong'  # small float diff
    if 1.1 <= ratio <= 2.0:
        return 'slightly_larger'
    if 2.0 < ratio <= 10.0:
        return 'much_larger'
    if ratio > 10.0:
        return 'way_larger'
    # fallback: check for intermediate keywords near extracted number
    if re.search(r'\b(total|subtotal|cost|remaining|so far|combined|sum|profit|revenue)\b', snip, re.IGNORECASE):
        return 'picked_intermediate'
    return 'other'


def main():
    cats = collections.Counter()
    samples = collections.defaultdict(list)
    lines = []
    with open(IN_PATH, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue
        filepart, gt, ex, snip = parsed
        cat = categorize(gt, ex, snip)
        cats[cat] += 1
        if len(samples[cat]) < 5:
            samples[cat].append((filepart, gt, ex, snip))
    # print summary
    print('Total lines analyzed:', len(lines))
    print('\nCategory counts:')
    for k,v in cats.most_common():
        print(f'{k}: {v}')
    print('\nSample examples per category (up to 5 each):')
    for k in cats:
        print('\n==', k, '==')
        for f,gt,ex,sn in samples[k]:
            snippet_display = sn[:140].replace('\n', ' ')
            print(f'- {f} | GT:{gt} | EX:{ex} | Snippet start: {snippet_display}')

if __name__ == "__main__":
    main()
