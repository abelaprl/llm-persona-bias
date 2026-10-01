"""CLI parsing GSM8K Religion Persona — dipanggil sekali per model (K1b).

Contoh:
  python -m src.benchmarks.gsm8k.Religion_Persona.run --model mistralai_ministral-8b-2512
  python -m src.benchmarks.gsm8k.Religion_Persona.run --all --rules core
"""
import argparse
import csv
import importlib
import os
import sys
import time

from .._core import runner
from .personas import PERSONA_GROUP, PERSONAS, MODELS

RULE_PACK_BY_MODEL = {
    'deepseek_deepseek-r1-distill-qwen-32b': 'rules_deepseek_r1_distill_32b',
    'meta-llama_llama-3.1-8b-instruct': 'rules_llama31_8b',
    'meta-llama_llama-3.3-70b-instruct': 'rules_llama33_70b',
    'mistralai_ministral-8b-2512': 'rules_ministral_8b',
    'openai_gpt-oss-20b': 'rules_gpt_oss_20b',
    'qwen_qwen3-next-80b-a3b-thinking': 'rules_qwen3_next_80b',
}

DEFAULT_INPUT = r'result\GMS8K\Religion_Persona'
DEFAULT_OUT = r'result\GMS8K\reports_parser_v2'


def load_rules(model_slug, which):
    mod_name = 'rules_core' if which == 'core' else RULE_PACK_BY_MODEL[model_slug]
    return importlib.import_module(f'.{mod_name}', package=__package__)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', help='slug model; kosongkan bila pakai --all')
    ap.add_argument('--all', action='store_true', help='jalankan 6 model')
    ap.add_argument('--rules', default='model', choices=['model', 'core'],
                    help='"model" = rule pack milik model; "core" = pembanding identik')
    ap.add_argument('--input_root', default=DEFAULT_INPUT)
    ap.add_argument('--out_root', default=DEFAULT_OUT)
    ap.add_argument('--personas', help='daftar persona dipisah koma (default: 16)')
    ap.add_argument('--limit', type=int, help='batasi jumlah soal per persona (uji cepat)')
    ap.add_argument('--blank_ground_truth', action='store_true',
                    help='uji V2: kosongkan ground truth sebelum parsing')
    ap.add_argument('--no_split', action='store_true', help='lewati results_<persona>.csv')
    ap.add_argument('--reparse_results',
                    help='reparse cepat dari results.csv yang sudah memuat content utuh')
    ap.add_argument('--refresh_manifest',
                    help='manifest selective rerun; refresh target dari raw JSON sebelum reparse')
    args = ap.parse_args(argv)

    models = MODELS if args.all else [args.model]
    if not models or models == [None]:
        ap.error('pilih --model atau --all')
    personas = args.personas.split(',') if args.personas else PERSONAS

    for model_slug in models:
        rules = load_rules(model_slug, args.rules)
        suffix = '_core' if args.rules == 'core' else ''
        out_dir = os.path.join(args.out_root, PERSONA_GROUP, model_slug)
        print(f'\n{"=" * 78}\n  {model_slug}   rule_pack={rules.RULE_PACK}\n{"=" * 78}')

        t0 = time.time()
        if args.reparse_results:
            refresh_files = set()
            if args.refresh_manifest:
                prefix = f'{PERSONA_GROUP}/{model_slug}/'
                with open(args.refresh_manifest, encoding='utf-8-sig', newline='') as mf:
                    for manifest_row in csv.DictReader(mf):
                        value = (manifest_row.get('source_json') or
                                 manifest_row.get('relative_path') or '')
                        value = value.replace('\\', '/').lstrip('/')
                        if value.startswith(prefix):
                            value = value[len(prefix):]
                        if value:
                            refresh_files.add(value)
            out = runner.reparse_results(
                source_results_path=args.reparse_results, rules=rules,
                out_dir=out_dir, blank_ground_truth=args.blank_ground_truth,
                results_filename=f'results{suffix}.csv',
                raw_root=os.path.join(args.input_root, model_slug),
                refresh_files=refresh_files)
        else:
            out = runner.parse_model(
                input_root=args.input_root, model_slug=model_slug,
                persona_group=PERSONA_GROUP, personas=personas, rules=rules,
                out_dir=out_dir, blank_ground_truth=args.blank_ground_truth,
                limit=args.limit, results_filename=f'results{suffix}.csv')

        acc_path, rows = runner.write_accuracy(
            out['stats'], model_slug, rules.RULE_PACK, out_dir,
            filename=f'accuracy{suffix}.csv')
        runner.write_coverage(out['stats'], model_slug, rules.RULE_PACK, out_dir,
                              filename=f'coverage{suffix}.csv')
        if not args.no_split and args.rules == 'model':
            runner.split_per_persona(out['results_path'], out_dir)

        overall = rows[-1] if rows else {}
        print(f'\n  ROWS={out["rows"]}  acc={overall.get("accuracy")}%  '
              f'commit={overall.get("commit_rate")}%  '
              f'candidate_only={overall.get("candidate_only")}  '
              f'({time.time() - t0:.1f}s)')
        print(f'  -> {out_dir}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
