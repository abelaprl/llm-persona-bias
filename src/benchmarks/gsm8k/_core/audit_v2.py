"""Generate extraction audit samples and technical-rerun manifest (no API)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MANIFEST_COLUMNS = [
    'model', 'persona', 'question_id', 'raw_file', 'failure_class',
    'response_success', 'finish_reason', 'provider_error_code',
    'provider_error_message', 'retry_count_existing',
    'eligible_for_technical_rerun', 'reason', 'trailing_content_present',
]


def _load_for_audit(path):
    text = path.read_text(encoding='utf-8')
    try:
        return json.loads(text), False, ''
    except json.JSONDecodeError as exc:
        try:
            obj, end = json.JSONDecoder().raw_decode(text.lstrip())
            trailing = bool(text.lstrip()[end:].strip())
            return obj, trailing, f'{type(exc).__name__}: {exc}'
        except Exception:
            return None, False, f'{type(exc).__name__}: {exc}'


def build_manifest(raw_root, output_path):
    rows = []
    for path in sorted(Path(raw_root).glob('*/*/gsm8k_*.json')):
        model, persona = path.parts[-3], path.parts[-2]
        data, trailing, json_error = _load_for_audit(path)
        qid = path.stem.replace('gsm8k_', '').lstrip('0') or '0'
        if not isinstance(data, dict):
            rows.append(dict.fromkeys(MANIFEST_COLUMNS, '') | {
                'model': model, 'persona': persona, 'question_id': qid,
                'raw_file': str(path), 'failure_class': 'invalid_json_unrecoverable',
                'eligible_for_technical_rerun': 'False',
                'reason': json_error, 'trailing_content_present': trailing})
            continue
        inp, resp = data.get('input') or {}, data.get('response') or {}
        qid = inp.get('question_id', qid)
        raw = resp.get('raw_api_response') or {}
        choices = raw.get('choices') or []
        finish = choices[0].get('finish_reason', '') if choices and isinstance(choices[0], dict) else ''
        success = resp.get('success') is True
        err = resp.get('error') or {}
        if isinstance(err, dict):
            code = err.get('code', '')
            message = err.get('message', '')
        else:
            code, message = '', str(err)
        perf = resp.get('performance') or {}
        eligible = (not success) or finish == 'error'
        failure = ('json_extra_data' if json_error and trailing else
                   'response_failure' if not success else
                   'finish_reason_error' if finish == 'error' else
                   'content_filter' if finish == 'content_filter' else
                   'length' if finish == 'length' else '')
        # Extra data sendiri bukan alasan rerun: object pertama harus menentukan
        # apakah ada kegagalan transport yang benar-benar eligible.
        if failure or json_error:
            rows.append({
                'model': model, 'persona': persona, 'question_id': qid,
                'raw_file': str(path), 'failure_class': failure or 'invalid_json',
                'response_success': success, 'finish_reason': finish,
                'provider_error_code': code, 'provider_error_message': str(message)[:500],
                'retry_count_existing': perf.get('retry_count', ''),
                'eligible_for_technical_rerun': eligible,
                'reason': ('transport/provider failure' if eligible else
                           'length excluded by methodology' if finish == 'length' else
                           'content_filter requires methodological decision' if finish == 'content_filter' else
                           'salvaged first JSON object; no technical failure'),
                'trailing_content_present': trailing,
            })
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS); w.writeheader(); w.writerows(rows)
    return rows


def build_extraction_audit(report_root, output_path, per_stratum=3):
    groups = defaultdict(list)
    for path in sorted(Path(report_root).glob('*/results.csv')):
        with path.open(encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                key = tuple(row.get(k, '') for k in
                            ('model', 'parse_source', 'priority_hit',
                             'extraction_confidence', 'finish_reason', 'persona'))
                if len(groups[key]) < per_stratum:
                    groups[key].append(row)
    rows = [row for key in sorted(groups) for row in groups[key]]
    fields = list(rows[0]) + ['audit_extracted_span', 'annotator_valid_without_gt', 'audit_notes'] if rows else []
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as fh:
        if fields:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', default=r'result/benchmarks/gsm8k\runs\main\Religion_Persona')
    ap.add_argument('--report_root', default=r'result/benchmarks/gsm8k\reports_parser_v2\Religion_Persona')
    ap.add_argument('--audit_root', default=r'result/benchmarks/gsm8k\audit')
    args = ap.parse_args()
    root = Path(args.audit_root)
    m = build_manifest(args.raw_root, root / 'religion_persona_technical_rerun_manifest.csv')
    a = build_extraction_audit(args.report_root, root / 'extraction_audit.csv')
    print(f'manifest_rows={len(m)} audit_rows={len(a)}')


if __name__ == '__main__':
    main()
