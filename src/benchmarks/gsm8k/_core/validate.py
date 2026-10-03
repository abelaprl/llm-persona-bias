"""Validasi tingkat data: V2 (no-gold-peek) dan V3 (differential measurement error).

  python -m src.benchmarks.gsm8k._core.validate --v2 --limit 200
  python -m src.benchmarks.gsm8k._core.validate --v3
"""
import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict

csv.field_size_limit(1 << 30)


# ── V2: hasil ekstraksi tidak boleh berubah ketika ground truth dikosongkan ──
def v2_no_gold_peek(input_root, out_root, group, models, personas, load_rules, limit):
    from . import runner
    print('\nV2 — no-gold-peek (data)')
    all_ok = True
    for model in models:
        rules = load_rules(model, 'model')
        real, blank = {}, {}
        for tag, blank_gt in (('real', False), ('blank', True)):
            with tempfile.TemporaryDirectory() as td:
                out = runner.parse_model(
                    input_root=input_root, model_slug=model, persona_group=group,
                    personas=personas, rules=rules, out_dir=td,
                    blank_ground_truth=blank_gt, limit=limit, progress=False)
                target = real if tag == 'real' else blank
                with open(out['results_path'], encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        target[row['file']] = (row['extracted_answer'],
                                               row['candidate_answer'],
                                               row['parse_source'])
        diff = [k for k in real if real[k] != blank.get(k)]
        ok = not diff
        all_ok &= ok
        print(f'  {"PASS" if ok else "FAIL"}  {model:<42} n={len(real)} beda={len(diff)}')
        for k in diff[:3]:
            print(f'        {k}: {real[k]} vs {blank.get(k)}')
    print(f'\n  V2 {"LULUS" if all_ok else "GAGAL"}')
    return all_ok


# ── V3: apakah tingkat abstain berasosiasi dengan persona ───────────────────
def _chi2(table):
    """Chi-square kemandirian. table: {baris: {kolom: hitung}}"""
    rows = list(table)
    cols = sorted({c for r in table.values() for c in r})
    n = sum(v for r in table.values() for v in r.values())
    if n == 0:
        return 0.0, 0, 1.0
    rt = {r: sum(table[r].values()) for r in rows}
    ct = {c: sum(table[r].get(c, 0) for r in rows) for c in cols}
    chi = 0.0
    for r in rows:
        for c in cols:
            e = rt[r] * ct[c] / n
            if e > 0:
                chi += (table[r].get(c, 0) - e) ** 2 / e
    df = (len(rows) - 1) * (len(cols) - 1)
    return chi, df, _chi2_sf(chi, df)


def _chi2_sf(x, k):
    """P(X > x) untuk chi-square df=k, tanpa scipy."""
    if k <= 0 or x <= 0:
        return 1.0
    if k % 2 == 0:                       # df genap: deret tertutup
        m, term, s = k // 2, 1.0, 1.0
        for i in range(1, m):
            term *= (x / 2) / i
            s += term
        return min(1.0, math.exp(-x / 2) * s)
    # df ganjil: lewat fungsi gamma tak lengkap (deret)
    a, xx = k / 2.0, x / 2.0
    term, s, n = 1.0 / a, 1.0 / a, 0
    while n < 1000:
        n += 1
        term *= xx / (a + n)
        s += term
        if term < 1e-14 * s:
            break
    p = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    return max(0.0, min(1.0, 1.0 - p))


def v3_abstain_vs_persona(out_root, group, models, alpha=0.05):
    print('\nV3 — apakah tingkat abstain bergantung persona?')
    print(f'  alpha={alpha}, koreksi Bonferroni untuk {len(models)} model '
          f'-> ambang {alpha / len(models):.5f}\n')
    verdicts = {}
    for model in models:
        p = os.path.join(out_root, group, model, 'results.csv')
        if not os.path.exists(p):
            print(f'  [SKIP] {model}')
            continue
        table = defaultdict(lambda: defaultdict(int))
        with open(p, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                table[row['persona']][row['answer_status']] += 1
        chi, df, pv = _chi2({k: dict(v) for k, v in table.items()})
        sig = pv < alpha / len(models)
        verdicts[model] = (chi, df, pv, sig)
        print(f'  {model:<42} chi2={chi:8.2f} df={df:3d} p={pv:.3e} '
              f'{"SIGNIFIKAN" if sig else "-"}')
    flagged = [m for m, v in verdicts.items() if v[3]]
    if flagged:
        print('\n  Tingkat kegagalan ekstraksi BERASOSIASI dengan persona pada:')
        for m in flagged:
            print(f'    - {m}')
        print('  -> perbandingan akurasi antar persona untuk model itu wajib')
        print('     dilaporkan dengan peringatan (plan N5).')
    else:
        print('\n  Tidak ada model yang tingkat abstainnya bergantung persona.')

    out = os.path.join(out_root, group, '_combined')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'v3_abstain_vs_persona.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'chi2', 'df', 'p_value', 'alpha_bonferroni', 'significant'])
        for m, (chi, df, pv, sig) in verdicts.items():
            w.writerow([m, round(chi, 4), df, f'{pv:.6e}',
                        round(alpha / len(models), 6), sig])
    return verdicts


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', action='store_true')
    ap.add_argument('--v3', action='store_true')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--input_root', default=r'result/benchmarks/gsm8k\Religion_Persona')
    ap.add_argument('--out_root', default=r'result/benchmarks/gsm8k\reports')
    args = ap.parse_args()

    from ..Religion_Persona.personas import MODELS, PERSONAS, PERSONA_GROUP
    from ..Religion_Persona.run import load_rules

    if args.v2:
        v2_no_gold_peek(args.input_root, args.out_root, PERSONA_GROUP, MODELS,
                        PERSONAS, load_rules, args.limit)
    if args.v3:
        v3_abstain_vs_persona(args.out_root, PERSONA_GROUP, MODELS)
