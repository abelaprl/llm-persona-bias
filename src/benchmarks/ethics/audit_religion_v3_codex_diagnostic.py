"""Prepare a development-only AI diagnostic review; never score ETHICS answers."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv'
SPLIT = ROOT / 'result/ETHICS/reports/religion_lm_v3_partition_revision_20260915_r1/question_split.csv'
QUOTAS = {'commonsense': 3, 'deontology': 3, 'justice': 2, 'virtue': 2}
SEED = 18222008


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    with SPLIT.open(encoding='utf-8-sig', newline='') as handle:
        partitions = {(r['subset'], r['question_id']): r['partition'] for r in csv.DictReader(handle)}
    counts, reservoirs, generators = Counter(), {}, {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            subset, question = row['subset'], row['question_id']
            if partitions[(subset, question)] != 'development':
                continue
            final = row['response_content_full']
            surface = prepare_final_surface(final)
            if not surface.explanation_available:
                continue
            cell = (row['model'], subset)
            if cell not in generators:
                generators[cell] = random.Random(int(hashlib.sha256(f'{SEED}|{cell}'.encode()).hexdigest(), 16))
                reservoirs[cell] = []
            counts[cell] += 1
            item = {'model_id': row['model'], 'persona': row['persona_folder'], 'subset': subset,
                    'question_id': question, 'final_input': final}
            n = QUOTAS[subset]
            if len(reservoirs[cell]) < n:
                reservoirs[cell].append(item)
            else:
                index = generators[cell].randrange(counts[cell])
                if index < n:
                    reservoirs[cell][index] = item
    records = []
    for cell, items in sorted(reservoirs.items()):
        for item in sorted(items, key=lambda r: (r['persona'], r['question_id'])):
            raw = item.pop('final_input')
            surface = prepare_final_surface(raw)
            summary, hits = extract_final(raw)
            item.update(audit_id=f'A{len(records)+1:03d}', reviewer_type='ai', partition='development',
                        response_key=json.dumps([item[k] for k in ['model_id', 'persona', 'subset', 'question_id']], separators=(',', ':')),
                        raw_final_sha256=summary['raw_final_sha256'], final_text=surface.visible_text,
                        word_count=summary['word_count'], surface_flags=summary['surface_flags'],
                        inclusion_probability=QUOTAS[cell[1]] / counts[cell], occurrences=hits)
            records.append(item)
    assert len(records) == 60
    write_json(out / 'sample_full.json', records)
    manifest = {'status': 'prepared_ai_diagnostic_not_gold', 'reviewer_type': 'ai', 'seed': SEED,
                'source': str(SOURCE), 'source_sha256': digest(SOURCE), 'split': str(SPLIT),
                'split_sha256': digest(SPLIT), 'sample_sha256': digest(out / 'sample_full.json'),
                'responses': len(records), 'partition': 'development', 'eligibility': 'linguistic_text',
                'sampling': 'independent_reservoir_per_model_subset', 'quotas_per_model': QUOTAS,
                'eligible_counts': [{'model_id': m, 'subset': s, 'eligible': n} for (m, s), n in sorted(counts.items())],
                'human_reviewers_completed': 0, 'promoted_rules': [],
                'script_sha256': digest(Path(__file__))}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'responses': len(records), 'accepted_occurrences': sum(h['candidate_status'] != 'excluded' for r in records for h in r['occurrences']), 'out': str(out)}))


def show(out, start, end):
    for record in json.loads((out / 'sample_full.json').read_text(encoding='utf-8'))[start-1:end]:
        print('\n' + record['audit_id'] + ' ' + record['model_id'] + ' ' + record['persona'] + ' ' + record['subset'])
        print(record['final_text'])
        print('ACCEPTED SPANS:')
        for hit in record['occurrences']:
            if hit['candidate_status'] != 'excluded':
                print(f"{hit['marker_id']} [{hit['raw_start']}:{hit['raw_end']}]: {hit['raw_matched_text']!r}")


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    records = json.loads((out / 'sample_full.json').read_text(encoding='utf-8'))
    reviews = json.loads((out / 'reviews.json').read_text(encoding='utf-8'))
    assert digest(SOURCE) == manifest['source_sha256']
    assert digest(SPLIT) == manifest['split_sha256']
    assert digest(out / 'sample_full.json') == manifest['sample_sha256']
    assert len(records) == len(reviews) == 60
    assert len({r['response_key'] for r in records}) == 60
    assert {r['audit_id'] for r in records} == {r['audit_id'] for r in reviews}
    assert len({r['audit_id'] for r in reviews}) == 60
    assert all(r['reviewer_type'] == 'ai' and r['full_response_read'] is True and r['notes'] for r in reviews)
    selected = {r['response_key']: r for r in records}
    confirmed = 0
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = json.dumps([row[k] for k in ['model', 'persona_folder', 'subset', 'question_id']], separators=(',', ':'))
            if key not in selected:
                continue
            record = selected[key]
            raw = row['response_content_full']
            assert hashlib.sha256(raw.encode()).hexdigest() == record['raw_final_sha256']
            for hit in record['occurrences']:
                assert raw[hit['raw_start']:hit['raw_end']] == hit['raw_matched_text']
            confirmed += 1
    assert confirmed == 60
    with SPLIT.open(encoding='utf-8-sig', newline='') as handle:
        development = {(r['subset'], r['question_id']) for r in csv.DictReader(handle) if r['partition'] == 'development'}
    assert all((r['subset'], r['question_id']) in development for r in records)
    cells = Counter((r['model_id'], r['subset']) for r in records)
    assert len(cells) == 24 and all(n == QUOTAS[s] for (m, s), n in cells.items())
    result = {'status': 'ai_diagnostic_artifacts_verified_not_human_validation', 'responses': 60,
              'source_and_split_hashes_unchanged': True, 'span_provenance_verified': True,
              'development_only': True, 'review_records': len(reviews),
              'flagged_responses': sum(bool(r['issues']) for r in reviews),
              'review_sha256': digest(out / 'reviews.json')}
    write_json(out / 'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'show', 'verify'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=10)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.out)
    elif args.action == 'show':
        show(args.out, args.start, args.end)
    else:
        verify(args.out)
