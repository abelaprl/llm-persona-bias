"""Reproduce, audit, and report the Regional ETHICS candidate in a new run."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import io
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'result/ETHICS/reports'
SOURCE = REPORTS / 'Regional_Cultural_Persona/_combined/all_models_results_full.csv'
FROZEN = REPORTS / 'Regional_Cultural_Persona/linguistic_markers_regional_cultural_candidate_20260822_v1_0_16'
RECON = REPORTS / 'regional_lm_plan_recon_20260916_r1/full_response_readback.json'
EXPECTED_HASH = '09f96bec32feafe3426c39526c1d9a80453d09ede2edb06a25e90fbddacc89a5'
EXPECTED_ROWS = 154428
REPLAY_FILES = ('response_level.csv.gz', 'marker_hits.csv.gz', 'surface_texts.jsonl.gz',
                'registry.json', 'baseline_replay_diff.csv', 'old_new_delta.csv',
                'baseline_replay_summary.json')


def prepare_runtime() -> None:
    from src.benchmarks.ethics.run_gender_ethics_lm_workspace import enable_marker_imports
    enable_marker_imports()


def create_run(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def extraction_dependencies() -> dict:
    paths = [Path(__file__).parent / 'regional_lm_v3_extract.py',
             Path(__file__).parent / 'run_gender_ethics_lm_workspace.py',
             *sorted((ROOT / 'src/core/linguistic_markers').glob('*.py'))]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def replay_projection_hash() -> str:
    code = '\n'.join(inspect.getsource(fn) for fn in (equal_metric, _batches, replay))
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def seal_replay(run: Path) -> dict:
    """Record immutable artifact hashes after an independently verified full replay."""
    summary = json.loads((run / 'baseline_replay_summary.json').read_text(encoding='utf-8'))
    if not (summary.get('full_corpus') and summary.get('rows') == EXPECTED_ROWS
            and summary.get('unique_keys') == EXPECTED_ROWS and summary.get('baseline_difference_cells') == 0):
        raise ValueError('Only a complete zero-drift replay can be sealed')
    dependencies = extraction_dependencies()
    producer = json.loads((run / 'manifest_start.json').read_text(encoding='utf-8'))
    if producer['source_sha256'] != EXPECTED_HASH or any(
            producer['code_sha256'].get(key) != value for key, value in dependencies.items()):
        raise ValueError('Producer source/dependency hash mismatch')
    seal = dict(source_sha256=EXPECTED_HASH, dependency_sha256=dependencies,
                replay_projection_sha256=replay_projection_hash(),
                artifact_sha256={name: digest(run / name) for name in REPLAY_FILES})
    write_json(run / 'replay_artifacts.json', seal)
    return seal


def restore_replay(source_run: Path, out: Path) -> dict:
    """Validate every hash before copying a replay into a separate, empty run."""
    seal = json.loads((source_run / 'replay_artifacts.json').read_text(encoding='utf-8'))
    if seal.get('source_sha256') != EXPECTED_HASH or seal.get('dependency_sha256') != extraction_dependencies():
        raise ValueError('Replay source/dependency hash mismatch')
    if seal.get('replay_projection_sha256') != replay_projection_hash():
        raise ValueError('Replay serialization/projection hash mismatch')
    for name in REPLAY_FILES:
        if seal['artifact_sha256'].get(name) != digest(source_run / name):
            raise ValueError(f'Replay artifact hash mismatch: {name}')
        if (out / name).exists():
            raise FileExistsError(out / name)
    summary = json.loads((source_run / 'baseline_replay_summary.json').read_text(encoding='utf-8'))
    if not (summary.get('full_corpus') and summary.get('rows') == EXPECTED_ROWS
            and summary.get('unique_keys') == EXPECTED_ROWS and summary.get('baseline_difference_cells') == 0):
        raise ValueError('Replay is incomplete or has drift')
    for name in REPLAY_FILES:
        with (source_run / name).open('rb') as src, (out / name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    write_json(out / 'replay_artifacts.json', seal)
    write_json(out / 'replay_reuse.json', dict(source_run=str(source_run.resolve()),
                                              seal_sha256=digest(source_run / 'replay_artifacts.json'),
                                              interpretation='Exact hashed full replay reused; audit/report rebuilt.'))
    return dict(summary, reused_from=str(source_run.resolve()))


def verify() -> tuple[bool, str]:
    prepare_runtime()
    stream = io.StringIO()
    names = ['test_regional_cultural_ethics_linguistic_markers_v2', 'test_regional_lm_v3']
    for extra in ('test_regional_lm_v3_audit', 'test_regional_lm_v3_report'):
        if (Path(__file__).parent / 'tests' / f'{extra}.py').exists():
            names.append(extra)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(
        'src.benchmarks.ethics.tests.' + name) for name in names)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return result.wasSuccessful(), stream.getvalue()


def equal_metric(old: str, new: object) -> bool:
    if new is None:
        return old in ('', 'nan', 'NaN', 'None')
    if isinstance(new, bool):
        return old.lower() in ({'true', '1', '1.0'} if new else {'false', '0', '0.0'})
    if isinstance(new, (int, float)):
        try:
            if isinstance(new, float):
                # The frozen builder used pandas downcast='float' before CSV export.
                return np.float32(float(old)) == np.float32(new)
            return math.isclose(float(old), new, rel_tol=1e-12, abs_tol=1e-12)
        except ValueError:
            return False
    return old == str(new)


def subprocess_batch(texts: list[str]) -> list:
    """Use ordinary child stdio pipes; Windows multiprocessing named pipes are restricted."""
    completed = subprocess.run(
        [sys.executable, '-X', 'utf8', '-m', 'src.benchmarks.ethics.regional_lm_v3_extract'],
        input=json.dumps(texts, ensure_ascii=False), capture_output=True, text=True,
        encoding='utf-8', check=True, cwd=ROOT,
    )
    return json.loads(completed.stdout)


def _batches(source: Path, frozen: Path, limit: int | None):
    with source.open(encoding='utf-8-sig', newline='') as a, gzip.open(
            frozen / 'response_level_markers.csv.gz', 'rt', encoding='utf-8-sig', newline='') as b:
        pairs = itertools.zip_longest(csv.DictReader(a), csv.DictReader(b))
        batch = []
        for index, (raw, old) in enumerate(pairs):
            if limit is not None and index >= limit:
                break
            if raw is None or old is None:
                raise ValueError('Source and frozen row counts differ')
            left = (raw['model'], raw['persona_folder'], raw['question_id'])
            right = (old['model_id'], old['persona'], old['question_id'])
            if left != right:
                raise ValueError(f'Source and frozen ordering/key mismatch at row {index + 1}')
            batch.append((raw, old))
            if len(batch) == 512:
                yield batch
                batch = []
        if batch:
            yield batch


def replay(source: Path, frozen: Path, out: Path, *, workers: int = 4,
           limit: int | None = None) -> dict:
    from src.benchmarks.ethics.regional_lm_v3_extract import extract_batch, registry_snapshot, VERSION
    write_json(out / 'registry.json', registry_snapshot())
    counts, differences, candidate_differences = Counter(), Counter(), Counter()
    all_keys = set()
    hit_columns = ['response_key', 'marker_id', 'rule_id', 'surface', 'start', 'end', 'matched_text',
                   'decision', 'reason', 'rule_version', 'pattern_index', 'exclusion_rule_id']
    metadata = ('extraction_confidence', 'extracted_answer', 'is_correct', 'numeric_semantic_conflict',
                'parse_source', 'candidate_answer', 'candidate_source', 'abstain_reason', 'response_flags',
                'extracted_stance', 'construct_valid')
    started = time.monotonic()
    with gzip.open(out / 'response_level.csv.gz', 'xt', encoding='utf-8', newline='', compresslevel=3) as rf, \
         gzip.open(out / 'marker_hits.csv.gz', 'xt', encoding='utf-8', newline='', compresslevel=3) as hf, \
         gzip.open(out / 'surface_texts.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as sf, \
         (out / 'baseline_replay_diff.csv').open('x', encoding='utf-8', newline='') as df, \
         (out / 'old_new_delta.csv').open('x', encoding='utf-8', newline='') as cf:
        hits_writer = csv.DictWriter(hf, fieldnames=hit_columns)
        hits_writer.writeheader()
        diff_columns = ['response_key', 'column', 'stored', 'replayed']
        diff_writer, change_writer = (csv.DictWriter(f, fieldnames=diff_columns) for f in (df, cf))
        diff_writer.writeheader()
        change_writer.writeheader()
        response_writer = None

        def consume(batch, extracted):
            nonlocal response_writer
            for (raw, old), (base, candidate) in zip(batch, extracted):
                key = '|'.join((raw['model'], raw['persona_folder'], raw['question_id']))
                if key in all_keys:
                    raise ValueError(f'Duplicate response key: {key}')
                all_keys.add(key)
                for col, value in base.items():
                    if col in old and not equal_metric(old[col], value):
                        differences[col] += 1
                        diff_writer.writerow(dict(response_key=key, column=col, stored=old[col], replayed=value))
                result = dict(old)
                result.update(candidate['metrics'])
                for col, value in candidate['metrics'].items():
                    if col.endswith('__density') or col in ('avg_sentence_length', 'ttr_300'):
                        result[col] = np.float32(value) if value is not None else None
                result.update({k: raw[k] for k in metadata})
                result.update(response_key=key, pipeline_version=VERSION,
                              raw_final_sha256=candidate['raw_final_sha256'],
                              answer_status=raw['answer_status'],
                              analysis_answer_status='abstain' if raw['extraction_confidence'] == 'semantic_unaudited'
                              else raw['answer_status'])
                for col, value in candidate['metrics'].items():
                    if col in old and not equal_metric(old[col], value):
                        candidate_differences[col] += 1
                        change_writer.writerow(dict(response_key=key, column=col, stored=old[col], replayed=value))
                if response_writer is None:
                    response_writer = csv.DictWriter(rf, fieldnames=list(result))
                    response_writer.writeheader()
                response_writer.writerow(result)
                for hit in candidate['hits']:
                    hits_writer.writerow(dict(response_key=key, **hit))
                    counts['hits_' + hit['decision']] += 1
                sf.write(json.dumps(dict(response_key=key, surfaces=candidate['surfaces'],
                                         raw_final_sha256=candidate['raw_final_sha256'],
                                         surface_sha256=candidate['surface_sha256']), ensure_ascii=False) + '\n')
                counts['rows'] += 1
            if counts['rows'] % 4096 == 0:
                print(f"[replay] {counts['rows']:,} rows; {time.monotonic() - started:.1f}s", flush=True)

        batches = iter(_batches(source, frozen, limit))
        if workers == 1:
            for batch in batches:
                consume(batch, extract_batch([raw['response_content_full'] for raw, _ in batch]))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = deque()
                for batch in batches:
                    pending.append((batch, executor.submit(subprocess_batch, [r['response_content_full'] for r, _ in batch])))
                    if len(pending) >= workers * 2:
                        ready, future = pending.popleft()
                        consume(ready, future.result())
                while pending:
                    ready, future = pending.popleft()
                    consume(ready, future.result())
    summary = dict(counts, baseline_difference_cells=sum(differences.values()),
                   baseline_difference_columns=dict(differences), candidate_difference_columns=dict(candidate_differences),
                   elapsed_seconds=time.monotonic() - started, full_corpus=limit is None,
                   unique_keys=len(all_keys), surface_replay_scope='stored status/counts/opening/polarity; original bundle does not store surface strings')
    summary['numeric_comparison'] = 'float32 roundtrip equality for stored float metrics; exact integer/bool/string equality'
    write_json(out / 'baseline_replay_summary.json', summary)
    if differences:
        raise ValueError('Baseline drift detected; review baseline_replay_diff.csv before continuing')
    if limit is None and counts['rows'] != EXPECTED_ROWS:
        raise ValueError(f'Unexpected full-corpus count: {counts["rows"]}')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('verify', 'prepare', 'all'), default='all')
    parser.add_argument('--run-id', default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Explicit smoke run; never release-eligible')
    parser.add_argument('--audit-input', type=Path)
    parser.add_argument('--replay-from', type=Path, help='Reuse only a sealed full replay with identical extraction hashes')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('run-id must contain only letters, numbers, underscore or hyphen')
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    if args.replay_from and args.limit:
        parser.error('--replay-from cannot be combined with --limit')
    prepare_runtime()
    ok, log = verify()
    print(log, flush=True)
    if not ok:
        return 1
    if args.stage == 'verify':
        return 0
    source_hash = digest(SOURCE)
    if source_hash != EXPECTED_HASH:
        raise ValueError('Canonical source hash differs from the frozen specification')
    out = REPORTS / ('regional_lm_v3_' + args.run_id)
    create_run(out)
    (out / 'baseline_test_log.txt').write_text(log, encoding='utf-8')
    legacy = REPORTS / 'Regional_Cultural_Persona/linguistic_markers/analysis/regional_lm_v2_0_1_complete_20260828'
    protected = [SOURCE, *sorted(p for p in FROZEN.rglob('*') if p.is_file()),
                 *sorted(p for p in legacy.rglob('*') if p.is_file())]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    code_paths = [*Path(__file__).parent.glob('regional_lm_v3_*.py'), Path(__file__),
                  *Path(__file__).parent.glob('tests/test_regional_lm_v3*.py'),
                  ROOT / 'src/benchmarks/ethics/run_gender_ethics_lm_workspace.py',
                  ROOT / 'src/benchmarks/ethics/regional_cultural_ethics_linguistic_markers_v2.py',
                  *sorted((ROOT / 'src/core/linguistic_markers').glob('*.py'))]
    manifest = dict(command=sys.argv, source=str(SOURCE), source_sha256=source_hash,
                    python=sys.version, seed=18222008, full_corpus=args.limit is None,
                    versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy')},
                    code_sha256={str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                    protected_sha256_before=before, audit_input=str(args.audit_input) if args.audit_input else None,
                    pipeline_status='running', tasks={})
    write_json(out / 'manifest_start.json', manifest)
    try:
        if args.replay_from:
            manifest['tasks']['replay'] = restore_replay(args.replay_from, out)
        else:
            manifest['tasks']['replay'] = replay(SOURCE, FROZEN, out, workers=args.workers, limit=args.limit)
            if args.limit is None:
                seal_replay(out)
        from src.benchmarks.ethics.regional_lm_v3_audit import inventory, make_audit_packets, summarize_audits
        manifest['tasks']['inventory'] = inventory(SOURCE, FROZEN, out)
        manifest['tasks']['packets'] = make_audit_packets(SOURCE, RECON, out)
        if args.audit_input:
            from src.benchmarks.ethics.regional_lm_v3_audit import import_audit
            manifest['tasks']['audit_import'] = import_audit(args.audit_input, out)
        else:
            manifest['tasks']['audit_summary'] = summarize_audits(out)
        if args.stage == 'all':
            from src.benchmarks.ethics.regional_lm_v3_report import build_report
            manifest['tasks']['report'] = build_report(out, FROZEN)
        manifest['protected_sha256_after'] = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        manifest['protected_inputs_unchanged'] = before == manifest['protected_sha256_after']
        if not manifest['protected_inputs_unchanged']:
            raise RuntimeError('Protected source artifacts changed during this run')
        manifest['pipeline_status'] = 'complete'
    except BaseException as error:
        manifest['pipeline_status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(out / 'manifest.json', manifest)
    print(f'[complete] {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
