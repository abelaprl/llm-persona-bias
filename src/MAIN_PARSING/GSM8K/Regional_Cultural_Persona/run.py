"""CLI parser GSM8K Regional Cultural Persona, satu eksekusi per model."""

import argparse
import csv
import importlib
import os
import sys
import time

from .._core import runner
from .personas import MODELS, PERSONAS, PERSONA_GROUP


RULE_PACK_BY_MODEL = {
    'deepseek_deepseek-r1-distill-qwen-32b': 'rules_deepseek_r1_distill_32b',
    'meta-llama_llama-3.1-8b-instruct': 'rules_llama31_8b',
    'meta-llama_llama-3.3-70b-instruct': 'rules_llama33_70b',
    'mistralai_ministral-8b-2512': 'rules_ministral_8b',
    'openai_gpt-oss-20b': 'rules_gpt_oss_20b',
    'qwen_qwen3-next-80b-a3b-thinking': 'rules_qwen3_next_80b',
}

DEFAULT_INPUT = r'result\GMS8K\regional_cultural_persona'
DEFAULT_OUT = r'result\GMS8K\reports'


def load_rules(model_slug, which):
    module = 'rules_core' if which == 'core' else RULE_PACK_BY_MODEL[model_slug]
    return importlib.import_module(f'.{module}', package=__package__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS,
                        help='slug model; kosongkan bila memakai --all')
    parser.add_argument('--all', action='store_true', help='jalankan enam model utama')
    parser.add_argument('--rules', default='model', choices=['model', 'core'])
    parser.add_argument('--input_root', default=DEFAULT_INPUT)
    parser.add_argument('--out_root', default=DEFAULT_OUT)
    parser.add_argument('--personas', help='daftar persona dipisah koma')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--read_workers', type=int, default=64,
                        help='jumlah worker pembaca JSON; urutan CSV tetap deterministik')
    parser.add_argument('--blank_ground_truth', action='store_true')
    parser.add_argument('--no_split', action='store_true')
    parser.add_argument('--reparse_results')
    parser.add_argument('--refresh_manifest')
    args = parser.parse_args(argv)

    models = MODELS if args.all else [args.model]
    if not models or models == [None]:
        parser.error('pilih --model atau --all')
    if args.read_workers < 1:
        parser.error('--read_workers minimal 1')
    personas = args.personas.split(',') if args.personas else PERSONAS
    unknown = sorted(set(personas) - set(PERSONAS))
    if unknown:
        parser.error(f'persona tidak dikenal: {", ".join(unknown)}')

    for model_slug in models:
        rules = load_rules(model_slug, args.rules)
        suffix = '_core' if args.rules == 'core' else ''
        out_dir = os.path.join(args.out_root, PERSONA_GROUP, model_slug)
        print(f'\n{"=" * 78}\n  {model_slug}   rule_pack={rules.RULE_PACK}\n{"=" * 78}')
        started = time.time()

        if args.reparse_results:
            refresh_files = set()
            if args.refresh_manifest:
                prefixes = (
                    f'{PERSONA_GROUP}/{model_slug}/',
                    f'regional_cultural_persona/{model_slug}/',
                )
                with open(args.refresh_manifest, encoding='utf-8-sig', newline='') as handle:
                    for row in csv.DictReader(handle):
                        value = (row.get('source_json') or row.get('relative_path') or '')
                        value = value.replace('\\', '/').lstrip('/')
                        for prefix in prefixes:
                            if value.startswith(prefix):
                                value = value[len(prefix):]
                                break
                        if value:
                            refresh_files.add(value)
            out = runner.reparse_results(
                source_results_path=args.reparse_results,
                rules=rules,
                out_dir=out_dir,
                blank_ground_truth=args.blank_ground_truth,
                results_filename=f'results{suffix}.csv',
                raw_root=os.path.join(args.input_root, model_slug),
                refresh_files=refresh_files,
                read_workers=args.read_workers,
            )
        else:
            out = runner.parse_model(
                input_root=args.input_root,
                model_slug=model_slug,
                persona_group=PERSONA_GROUP,
                personas=personas,
                rules=rules,
                out_dir=out_dir,
                blank_ground_truth=args.blank_ground_truth,
                limit=args.limit,
                results_filename=f'results{suffix}.csv',
                read_workers=args.read_workers,
            )

        _, accuracy_rows = runner.write_accuracy(
            out['stats'], model_slug, rules.RULE_PACK, out_dir,
            filename=f'accuracy{suffix}.csv',
        )
        runner.write_coverage(
            out['stats'], model_slug, rules.RULE_PACK, out_dir,
            filename=f'coverage{suffix}.csv',
        )
        if not args.no_split and args.rules == 'model':
            runner.split_per_persona(out['results_path'], out_dir)

        overall = accuracy_rows[-1] if accuracy_rows else {}
        print(
            f'\n  ROWS={out["rows"]}  acc={overall.get("accuracy")}%  '
            f'commit={overall.get("commit_rate")}%  '
            f'candidate_only={overall.get("candidate_only")}  '
            f'({time.time() - started:.1f}s)'
        )
        print(f'  -> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
