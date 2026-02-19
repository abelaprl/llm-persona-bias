import csv
from collections import Counter
import statistics

PATH = 'scripts/truncated_token_correlation.csv'

def main():
    finishes = Counter()
    totals = []
    rows = 0
    with open(PATH, encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows += 1
            fr = row.get('finish_reason')
            if fr:
                finishes[fr] += 1
            t = row.get('total_tokens')
            try:
                totals.append(int(t))
            except:
                pass

    print(f'processed rows: {rows}')
    print('finish_reason counts:')
    for k, v in finishes.most_common():
        print(f'  {k}: {v}')

    if totals:
        print('total_tokens min/median/mean/max =', min(totals), int(statistics.median(totals)), int(statistics.mean(totals)), max(totals))

if __name__ == '__main__':
    main()
