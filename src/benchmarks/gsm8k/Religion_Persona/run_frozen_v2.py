"""Hash-locked CLI for the isolated 2026-08-13 10:31:12 parser snapshot."""
import hashlib
import json
from pathlib import Path

from .._core import frozen_v2_skeleton, runner
from . import run

ROOT = Path(__file__).resolve().parents[4]
LOCK = Path(__file__).with_name('frozen_v2_source_lock.json')


def verify_source_lock():
    manifest = json.loads(LOCK.read_text(encoding='utf-8'))
    bad = []
    for rel, expected in manifest['sha256'].items():
        path = ROOT / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            bad.append(f'{rel}: expected {expected}, got {actual}')
    if bad:
        raise RuntimeError('Frozen-v2 source lock failed:\n' + '\n'.join(bad))
    return manifest


def main(argv=None):
    verify_source_lock()
    runner.extract = frozen_v2_skeleton.extract
    run.RULE_PACK_BY_MODEL['meta-llama_llama-3.3-70b-instruct'] = \
        'rules_llama33_70b_frozen_v2'
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if '--out_root' not in argv:
        argv = [*argv, '--out_root', r'result/benchmarks/gsm8k\reports_parser_frozen_v2']
    return run.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
