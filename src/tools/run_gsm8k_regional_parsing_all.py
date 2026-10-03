"""Run six GSM8K Regional Cultural parsers with bounded model concurrency.

Each model owns a different output directory, so model-level parallelism does
not create competing writers. In practice, concurrent full-model scans compete
for the same disk, so the safe and faster default is one model at a time with
parallel JSON reads inside that model. CSV row order remains deterministic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.gsm8k.Regional_Cultural_Persona.personas import MODELS  # noqa: E402


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('harus minimal 1')
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_workers', type=positive, default=1)
    parser.add_argument('--read_workers', type=positive, default=64)
    parser.add_argument('--rules', choices=('model', 'core'), default='model')
    parser.add_argument('--input_root', default=r'result/benchmarks/gsm8k\regional_cultural_persona')
    parser.add_argument('--out_root', default=r'result/benchmarks/gsm8k\reports')
    parser.add_argument('--limit', type=positive)
    return parser.parse_args()


def run_model(model: str, args: argparse.Namespace) -> tuple[str, int, str]:
    command = [
        sys.executable,
        '-u',
        '-m',
        'src.benchmarks.gsm8k.Regional_Cultural_Persona.run',
        '--model', model,
        '--rules', args.rules,
        '--input_root', args.input_root,
        '--out_root', args.out_root,
        '--read_workers', str(args.read_workers),
    ]
    if args.limit:
        command.extend(['--limit', str(args.limit)])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        errors='replace',
        check=False,
    )
    return model, completed.returncode, completed.stdout


def main() -> int:
    args = parse_args()
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.model_workers, len(MODELS))) as pool:
        futures = {pool.submit(run_model, model, args): model for model in MODELS}
        for future in as_completed(futures):
            model, code, output = future.result()
            print(f'\n### {model} exit={code}\n{output}', flush=True)
            if code:
                failures.append(model)
    if failures:
        print(f'FAILED: {", ".join(sorted(failures))}', file=sys.stderr)
        return 1
    print('DONE: six Regional Cultural models parsed successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
