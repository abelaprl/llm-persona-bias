"""Isolated rescue parser for selected Religion-Persona abstentions.

The canonical reports are read-only inputs.  Existing committed rows and all
technical/provider failures are copied unchanged.  Only rows whose existing
``abstain_reason`` is explicitly allowed may be promoted in the separate
output report.
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from .._core import schema, skeleton
from .._core.numbers import answers_equal, find_numbers, normalize_number
from .personas import MODELS
from .run import load_rules


DEFAULT_SOURCE_ROOT = Path(r'result\GMS8K\reports\Religion_Persona')
DEFAULT_OUT_ROOT = Path(
    r'result\GMS8K\reports_rescue\20260816_religion_candidate_rescue_v1'
)
DEFAULT_MODELS = tuple(model for model in MODELS if not model.startswith('deepseek_'))
ELIGIBLE_REASONS = frozenset({'truncated', 'candidate_only', 'no_final_answer'})
RESCUE_VERSION = 'religion_candidate_rescue_v1'


def _as_bool(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _numeric(value):
    if value is None or str(value).strip() == '':
        return None
    return normalize_number(str(value).strip())


def choose_rescue(row, rules):
    """Return ``(value, method)`` using response text only, or ``(None, '')``.

    Ground truth is intentionally absent from this function.  The increasingly
    permissive order is recorded in ``method`` so downstream analysis can
    distinguish a promoted parser candidate from a last-token fallback.
    """
    if _as_bool(row.get('technical_failure')):
        return None, ''
    reason = (row.get('abstain_reason') or '').strip()
    if reason not in ELIGIBLE_REASONS:
        return None, ''

    existing = _numeric(row.get('candidate_answer'))
    if existing is not None:
        return existing, 'existing_candidate'

    content = row.get('response_content_full') or ''
    if not content.strip():
        return None, ''

    # For a rescue run the user explicitly accepts weak/truncated extraction,
    # so the normal finish_reason=length gate is intentionally omitted here.
    relaxed = skeleton.extract(content, rules, finish_reason=None)
    if relaxed.value is not None:
        return relaxed.value, 'relaxed_' + (relaxed.source or 'parser')
    if relaxed.candidate is not None:
        return relaxed.candidate, 'relaxed_' + (
            relaxed.candidate_source or 'candidate'
        ).replace(':', '_')

    values = find_numbers(content)
    if values:
        return values[-1], 'last_numeric_token'
    return None, ''


def apply_rescue(row, rules):
    """Return a copied row, plus manifest metadata when it was promoted."""
    value, method = choose_rescue(row, rules)
    if value is None:
        return dict(row), None

    rescued = dict(row)
    old_reason = rescued.get('abstain_reason') or ''
    gold = _numeric(rescued.get('ground_truth_answer'))
    correct = answers_equal(value, gold) if gold is not None else False
    rescued.update({
        'extracted_answer': value,
        'answer_status': 'committed',
        'parse_source': 'rescue_parser',
        'priority_hit': method,
        'extraction_confidence': 'candidate_accepted',
        'abstain_reason': '',
        'is_correct': correct,
        'parser_version': RESCUE_VERSION,
    })
    manifest = {
        'model': rescued.get('model') or '',
        'persona': rescued.get('persona') or '',
        'file': rescued.get('file') or '',
        'original_abstain_reason': old_reason,
        'original_candidate_answer': row.get('candidate_answer') or '',
        'rescued_answer': value,
        'rescue_method': method,
        'is_correct': correct,
        'finish_reason': rescued.get('finish_reason') or '',
        'content_empty': rescued.get('content_empty') or '',
    }
    return rescued, manifest


MANIFEST_COLUMNS = [
    'model', 'persona', 'file', 'original_abstain_reason',
    'original_candidate_answer', 'rescued_answer', 'rescue_method',
    'is_correct', 'finish_reason', 'content_empty',
]

COVERAGE_COLUMNS = [
    'model', 'rule_pack', 'persona', 'total', 'committed', 'commit_rate',
    'correct', 'accuracy', 'rescued', 'rescued_correct', 'unresolved',
    'unresolved_truncated', 'unresolved_candidate_only',
    'unresolved_no_final_answer', 'technical_failure',
]


def _atomic_writer(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent
    )
    os.close(fd)
    handle = open(temporary, 'w', encoding='utf-8', newline='')
    writer = csv.DictWriter(
        handle, fieldnames=fieldnames, extrasaction='ignore',
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    return handle, writer, Path(temporary)


def _record(stats, row, was_rescued=False):
    targets = [stats[row.get('persona') or ''], stats['OVERALL']]
    for counter in targets:
        counter['total'] += 1
        if row.get('answer_status') == 'committed':
            counter['committed'] += 1
        else:
            counter['unresolved'] += 1
            reason = row.get('abstain_reason') or ''
            if reason in ELIGIBLE_REASONS:
                counter['unresolved_' + reason] += 1
        if _as_bool(row.get('is_correct')):
            counter['correct'] += 1
            if was_rescued:
                counter['rescued_correct'] += 1
        if was_rescued:
            counter['rescued'] += 1
        if _as_bool(row.get('technical_failure')):
            counter['technical_failure'] += 1


def _coverage_rows(model, rule_pack, stats):
    personas = sorted(name for name in stats if name != 'OVERALL') + ['OVERALL']
    for persona in personas:
        counter = stats[persona]
        total = counter['total']
        yield {
            'model': model,
            'rule_pack': rule_pack,
            'persona': persona,
            'total': total,
            'committed': counter['committed'],
            'commit_rate': round(100 * counter['committed'] / total, 4),
            'correct': counter['correct'],
            'accuracy': round(100 * counter['correct'] / total, 4),
            'rescued': counter['rescued'],
            'rescued_correct': counter['rescued_correct'],
            'unresolved': counter['unresolved'],
            'unresolved_truncated': counter['unresolved_truncated'],
            'unresolved_candidate_only': counter['unresolved_candidate_only'],
            'unresolved_no_final_answer': counter['unresolved_no_final_answer'],
            'technical_failure': counter['technical_failure'],
        }


def rescue_model(source_path, out_dir, model, rules, execute=False):
    """Audit or write one model's isolated rescue report."""
    source_path = Path(source_path).resolve()
    out_dir = Path(out_dir).resolve()
    results_path = out_dir / 'results.csv'
    if source_path == results_path:
        raise ValueError('output rescue tidak boleh sama dengan results sumber')

    csv.field_size_limit(sys.maxsize)
    stats = defaultdict(lambda: defaultdict(int))
    manifests = []
    source_committed = 0
    eligible = 0
    output_handle = output_writer = temporary = None
    if execute:
        output_handle, output_writer, temporary = _atomic_writer(
            results_path, schema.COLUMNS
        )

    try:
        with source_path.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            missing = [c for c in schema.COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError('results sumber tidak memiliki kolom: ' + ', '.join(missing))
            for row in reader:
                if row.get('answer_status') == 'committed':
                    source_committed += 1
                if ((row.get('abstain_reason') or '') in ELIGIBLE_REASONS
                        and not _as_bool(row.get('technical_failure'))):
                    eligible += 1
                rescued, manifest = apply_rescue(row, rules)
                if manifest is not None:
                    manifests.append(manifest)
                _record(stats, rescued, was_rescued=manifest is not None)
                if output_writer is not None:
                    output_writer.writerow(rescued)

        if output_handle is not None:
            output_handle.close()
            output_handle = None
            os.replace(temporary, results_path)
            temporary = None

            manifest_path = out_dir / 'rescue_manifest.csv'
            handle, writer, temp_manifest = _atomic_writer(
                manifest_path, MANIFEST_COLUMNS
            )
            try:
                writer.writerows(manifests)
                handle.close()
                os.replace(temp_manifest, manifest_path)
            finally:
                if not handle.closed:
                    handle.close()
                if temp_manifest.exists():
                    temp_manifest.unlink()

            coverage_path = out_dir / 'coverage.csv'
            handle, writer, temp_coverage = _atomic_writer(
                coverage_path, COVERAGE_COLUMNS
            )
            try:
                writer.writerows(_coverage_rows(model, rules.RULE_PACK, stats))
                handle.close()
                os.replace(temp_coverage, coverage_path)
            finally:
                if not handle.closed:
                    handle.close()
                if temp_coverage.exists():
                    temp_coverage.unlink()
    finally:
        if output_handle is not None:
            output_handle.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()

    overall = stats['OVERALL']
    return {
        'model': model,
        'total': overall['total'],
        'source_committed': source_committed,
        'eligible': eligible,
        'rescued': overall['rescued'],
        'rescued_correct': overall['rescued_correct'],
        'unresolved': overall['unresolved'],
        'final_committed': overall['committed'],
        'coverage': round(100 * overall['committed'] / overall['total'], 4),
        'output': str(results_path) if execute else '',
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument(
        '--execute', action='store_true',
        help='tanpa flag ini hanya audit/dry-run; tidak menulis output',
    )
    args = parser.parse_args(argv)
    models = [value.strip() for value in args.models.split(',') if value.strip()]
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        parser.error('model di luar scope Religion non-DeepSeek: ' + ', '.join(unknown))

    for model in models:
        source = args.source_root / model / 'results.csv'
        if not source.exists():
            raise FileNotFoundError(source)
        rules = load_rules(model, 'model')
        summary = rescue_model(
            source, args.out_root / model, model, rules, execute=args.execute
        )
        mode = 'WRITE' if args.execute else 'DRY-RUN'
        print(
            f"{mode} {model}: total={summary['total']} "
            f"eligible={summary['eligible']} rescued={summary['rescued']} "
            f"rescued_correct={summary['rescued_correct']} "
            f"unresolved={summary['unresolved']} "
            f"coverage={summary['coverage']}%"
        )
        if summary['output']:
            print('  -> ' + summary['output'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
