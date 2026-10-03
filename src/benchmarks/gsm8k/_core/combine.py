"""Gabungkan hasil per model jadi tabel lintas-model.

Dijalankan setelah semua model selesai diparse.
  python -m src.benchmarks.gsm8k._core.combine --group Religion_Persona
"""
import argparse
import csv
import os

csv.field_size_limit(1 << 30)

LITE_COLUMNS = [
    'model', 'rule_pack', 'persona', 'question_id',
    'ground_truth_answer', 'extracted_answer', 'candidate_answer',
    'answer_status', 'parse_source', 'priority_hit', 'extraction_confidence',
    'abstain_reason', 'is_correct',
    'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens',
]


def combine(out_root, group, models):
    base = os.path.join(out_root, group)
    comb = os.path.join(base, '_combined')
    os.makedirs(comb, exist_ok=True)

    # results lite (tanpa teks respons penuh — agar berkas tetap bisa dibuka)
    lite_path = os.path.join(comb, 'all_models_results.csv')
    n = 0
    with open(lite_path, 'w', newline='', encoding='utf-8') as fo:
        w = csv.DictWriter(fo, fieldnames=LITE_COLUMNS, extrasaction='ignore')
        w.writeheader()
        for m in models:
            p = os.path.join(base, m, 'results.csv')
            if not os.path.exists(p):
                print(f'  [SKIP] {m}: results.csv belum ada')
                continue
            with open(p, encoding='utf-8') as fi:
                for row in csv.DictReader(fi):
                    w.writerow(row)
                    n += 1
    print(f'  all_models_results.csv  {n} baris')

    for fname, out_name in [('accuracy.csv', 'all_models_accuracy.csv'),
                            ('accuracy_core.csv', 'all_models_accuracy_core.csv'),
                            ('coverage.csv', 'all_models_coverage.csv')]:
        rows, header = [], None
        for m in models:
            p = os.path.join(base, m, fname)
            if not os.path.exists(p):
                continue
            with open(p, encoding='utf-8') as fi:
                rd = csv.DictReader(fi)
                header = header or rd.fieldnames
                rows.extend(rd)
        if not rows:
            continue
        with open(os.path.join(comb, out_name), 'w', newline='', encoding='utf-8') as fo:
            w = csv.DictWriter(fo, fieldnames=header, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'  {out_name}  {len(rows)} baris')

    _ranking_check(comb)


def _ranking_check(comb):
    """Bandingkan peringkat model: rule pack sendiri vs rule pack core (plan Sec. 3.4)."""
    def overall(path):
        if not os.path.exists(path):
            return {}
        with open(path, encoding='utf-8') as f:
            return {r['model']: float(r['accuracy'])
                    for r in csv.DictReader(f) if r['persona'] == 'OVERALL'}

    a = overall(os.path.join(comb, 'all_models_accuracy.csv'))
    c = overall(os.path.join(comb, 'all_models_accuracy_core.csv'))
    if not a or not c:
        print('\n  [INFO] accuracy_core belum lengkap; uji peringkat dilewati.')
        return

    ra = [m for m, _ in sorted(a.items(), key=lambda kv: -kv[1])]
    rc = [m for m, _ in sorted(c.items(), key=lambda kv: -kv[1])]
    print('\n  Peringkat model (rule pack sendiri):')
    for i, m in enumerate(ra, 1):
        print(f'    {i}. {m:<42} {a[m]:6.2f}%')
    print('  Peringkat model (rule pack core, identik):')
    for i, m in enumerate(rc, 1):
        print(f'    {i}. {m:<42} {c[m]:6.2f}%')
    print(f'\n  Peringkat SAMA: {ra == rc}')
    if ra != rc:
        print('  -> Perbedaan peringkat WAJIB dilaporkan di naskah (plan N7).')

    out = os.path.join(comb, 'ranking_sensitivity.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'accuracy_rule_pack', 'rank_rule_pack',
                    'accuracy_core', 'rank_core', 'rank_shift'])
        for m in a:
            w.writerow([m, a[m], ra.index(m) + 1, c.get(m, ''),
                        rc.index(m) + 1 if m in rc else '',
                        (ra.index(m) - rc.index(m)) if m in rc else ''])
    print(f'  ranking_sensitivity.csv ditulis')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=r'result/benchmarks/gsm8k\reports')
    ap.add_argument('--group', default='Religion_Persona')
    args = ap.parse_args()
    if args.group == 'Regional_Cultural_Persona':
        from ..Regional_Cultural_Persona.personas import MODELS
    else:
        from ..Religion_Persona.personas import MODELS
    combine(args.out_root, args.group, MODELS)
