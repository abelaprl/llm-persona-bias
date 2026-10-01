"""Bind authored AI review decisions to source spans without changing extraction."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.benchmarks.ethics.audit_religion_v3_codex_diagnostic import SOURCE, digest, write_json
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface


def write_csv(path, records, columns=None):
    columns = columns or list(records[0])
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def finalize(out):
    sample = json.loads((out / 'sample_full.json').read_text(encoding='utf-8'))
    targeted = json.loads((out / 'targeted_full.json').read_text(encoding='utf-8'))
    reviews = json.loads((out / 'reviews.json').read_text(encoding='utf-8'))
    findings = json.loads((out / 'authored_findings.json').read_text(encoding='utf-8'))
    records = {r['audit_id']: r for r in sample + targeted}
    keys = {r['response_key']: r['audit_id'] for r in records.values()}
    raw_by_id = {}
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = json.dumps([row[k] for k in ['model', 'persona_folder', 'subset', 'question_id']], separators=(',', ':'))
            if key in keys:
                raw_by_id[keys[key]] = row['response_content_full']
    assert len(raw_by_id) == len(records) == 62
    bound = []
    for finding in findings:
        for evidence in finding['evidence']:
            audit_id = evidence['audit_id']
            record, raw = records[audit_id], raw_by_id[audit_id]
            surface = prepare_final_surface(raw)
            hits = record.get('occurrences', record.get('conditional_hits'))
            spans = []
            for start in evidence.get('raw_starts', []):
                matching = [h for h in hits if h['marker_id'] == finding['marker_id'] and h['raw_start'] == start and h['candidate_status'] != 'excluded']
                assert len(matching) == 1, (finding['id'], audit_id, start)
                spans.append((start, matching[0]['raw_end'], matching[0]['rule_id']))
            for start in evidence.get('visible_starts', []):
                end = surface.visible_text.find('\n', start)
                if end < 0:
                    end = len(surface.visible_text)
                a, b = surface.raw_span('visible', start, end)
                spans.append((a, b, ''))
            for quote in evidence.get('visible_quotes', []):
                assert surface.visible_text.count(quote) == 1, (audit_id, quote)
                start = surface.visible_text.index(quote)
                a, b = surface.raw_span('visible', start, start + len(quote))
                spans.append((a, b, ''))
            for a, b, rule_id in spans:
                if finding['kind'].startswith('false_negative'):
                    assert not any(h['marker_id'] == finding['marker_id'] and h['candidate_status'] != 'excluded' and h['raw_start'] < b and h['raw_end'] > a for h in hits), (finding['id'], audit_id, a)
                model, persona, subset, question_id = json.loads(record['response_key'])
                bound.append({'finding_id': finding['id'], 'audit_id': audit_id,
                              'sample_type': 'targeted' if audit_id.startswith('T') else 'random_stratified_development_text',
                              'reviewer_type': 'ai', 'decision': finding['kind'],
                              'marker_id': finding['marker_id'], 'rule_id': rule_id,
                              'model_id': model, 'persona': persona, 'subset': subset, 'question_id': question_id,
                              'response_key': record['response_key'], 'raw_final_sha256': record['raw_final_sha256'],
                              'raw_start': a, 'raw_end': b, 'evidence_text': raw[a:b], 'reason': finding['reason']})
    fp_keys = {(x['audit_id'], x['marker_id'], x['raw_start']) for x in bound if x['decision'] == 'false_positive'}
    occurrence_reviews = []
    for r in sample:
        for h in r['occurrences']:
            if h['candidate_status'] == 'excluded':
                continue
            occurrence_reviews.append({'audit_id': r['audit_id'], 'response_key': r['response_key'],
                                       'raw_final_sha256': r['raw_final_sha256'], 'marker_id': h['marker_id'],
                                       'rule_id': h['rule_id'], 'raw_start': h['raw_start'], 'raw_end': h['raw_end'],
                                       'matched_text': h['raw_matched_text'], 'reviewer_type': 'ai',
                                       'ai_decision': 'false_positive' if (r['audit_id'], h['marker_id'], h['raw_start']) in fp_keys else 'no_issue_identified_in_diagnostic_pass',
                                       'human_gold_status': 'not_available'})
    write_csv(out / 'finding_spans.csv', bound)
    write_csv(out / 'occurrence_reviews.csv', occurrence_reviews)
    write_json(out / 'reviewed_final_sources.json', [{'audit_id': a, 'response_key': records[a]['response_key'],
                'raw_final_sha256': records[a]['raw_final_sha256'], 'final_text': t} for a, t in sorted(raw_by_id.items())])
    by_rule = Counter(r['rule_id'] for r in occurrence_reviews)
    fp_rule = Counter(r['rule_id'] for r in occurrence_reviews if r['ai_decision'] == 'false_positive')
    response_rules = defaultdict(set)
    for r in occurrence_reviews:
        response_rules[r['rule_id']].add(r['audit_id'])
    quota_path = out.parent / 'religion_lm_v3_candidate_20260916_r4_overall/audit/rule_quotas.csv'
    with quota_path.open(encoding='utf-8-sig', newline='') as handle:
        quotas = list(csv.DictReader(handle))
    rule_rows = [{'rule_id': r['rule_id'], 'total_corpus_hits': int(r['total_rule_hits']),
                  'ai_reviewed_occurrences': by_rule[r['rule_id']],
                  'ai_reviewed_responses': len(response_rules[r['rule_id']]),
                  'ai_false_positive_occurrences': fp_rule[r['rule_id']] if by_rule[r['rule_id']] else '',
                  'human_reviewed': 0, 'human_false_positives': '',
                  'promotion_status': 'not_promoted',
                  'audit_status': 'diagnostic_only' if by_rule[r['rule_id']] else 'not_observed_in_sample'} for r in quotas]
    assert set(by_rule).issubset({r['rule_id'] for r in rule_rows})
    write_csv(out / 'rule_review_summary.csv', rule_rows)
    finding_counts = Counter((r['finding_id'], r['sample_type']) for r in bound)
    summary = {'status': 'ai_diagnostic_completed_validation_incomplete',
               'random_responses_read': 60, 'targeted_responses_read': 2,
               'models': len({r['model_id'] for r in sample}), 'personas': len({r['persona'] for r in sample}),
               'subsets': len({r['subset'] for r in sample}), 'linguistic_words_read_random': sum(r['word_count'] for r in sample),
               'accepted_occurrences_reviewed': len(occurrence_reviews),
               'markers_with_hits': len({r['marker_id'] for r in occurrence_reviews}),
               'rules_with_hits': len(by_rule), 'rules_in_inventory': len(rule_rows),
               'ai_false_positive_occurrences_random': sum(fp_rule.values()),
               'ai_false_positive_occurrences_targeted': sum(r['decision'] == 'false_positive' and r['sample_type'] == 'targeted' for r in bound),
               'ai_documented_missing_occurrences_random': sum(r['decision'].startswith('false_negative') and r['sample_type'] != 'targeted' for r in bound),
               'response_count_with_documented_issues': sum(bool(r['issues']) for r in reviews),
               'human_reviewers_completed': 0, 'promoted_rules': [],
               'precision': None, 'recall': None, 'reason_metrics_not_estimated': 'Development diagnostic, one AI reviewer, incomplete FN enumeration, clustered unequal sampling; no human gold.',
               'test_command': r'.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s src/benchmarks/ethics/tests -p test_religion*v3*.py -v',
               'test_result_observed': {'tests': 50, 'passed': 50, 'seconds': 68.111},
               'finding_counts': [{'finding_id': fid, 'sample_type': group, 'occurrences': count} for (fid, group), count in sorted(finding_counts.items())],
               'input_hashes': {p.name: digest(p) for p in [out / 'sample_full.json', out / 'reviews.json', out / 'authored_findings.json', quota_path]},
               'old_results_rewritten': False}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    finalize(parser.parse_args().out)
