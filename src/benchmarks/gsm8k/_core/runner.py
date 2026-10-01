"""Orkestrasi — jalan folder, panggil rangka + rule pack, tulis CSV inkremental.

Lapisan ini boleh berbeda per grup persona (K1b); ia tidak pernah menyentuh
nilai jawaban.
"""
import csv
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import schema
from .skeleton import extract
from .scoring import score
from .preprocess import get_content, has_separate_reasoning, looks_truncated


def _load_json_payload(path):
    """Load one JSON object, recovering only a complete object before garbage.

    A small number of interrupted writers left a valid, complete payload
    followed by extra bytes. ``raw_decode`` may recover that first object; an
    incomplete object is still rejected. The caller keeps ``strict_valid`` in
    the report so storage corruption is never hidden.
    """
    with open(path, encoding='utf-8-sig') as handle:
        text = handle.read()
    try:
        payload = json.loads(text)
        return payload, True, ''
    except json.JSONDecodeError as exc:
        stripped = text.lstrip()
        try:
            payload, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            raise exc
        if not stripped[end:].strip() or not isinstance(payload, dict):
            raise exc
        message = f'{type(exc).__name__}: {exc}'
        return payload, False, message


def _load_json_payload_safe(path):
    """Thread-pool friendly wrapper that keeps per-file errors as data."""
    try:
        payload, strict_valid, message = _load_json_payload(path)
        return payload, strict_valid, message, None
    except Exception as exc:  # reported in the corresponding CSV row
        return None, False, '', exc


def _write_csv_atomic(path, fieldnames, rows, quoting=csv.QUOTE_MINIMAL):
    """Write a complete CSV beside its target, then publish with os.replace."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f'.{os.path.basename(path)}.',
                                     suffix='.tmp', dir=directory)
    os.close(fd)
    try:
        with open(temp_path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore',
                               quoting=quoting)
            w.writeheader(); w.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _usage(resp):
    raw = resp.get('raw_api_response') or {}
    u = raw.get('usage') or {}
    ctd = u.get('completion_tokens_details') or {}
    return {
        'prompt_tokens': u.get('prompt_tokens') or 0,
        'completion_tokens': u.get('completion_tokens') or 0,
        'reasoning_tokens': ctd.get('reasoning_tokens') or 0,
        'total_tokens': u.get('total_tokens') or 0,
    }


def _finish_reason(resp):
    raw = resp.get('raw_api_response') or {}
    ch = raw.get('choices') or []
    return ch[0].get('finish_reason') if ch and isinstance(ch[0], dict) else None


def _response_error(resp):
    """Return provider error even when OpenRouter stores it on a choice."""
    err = resp.get('error')
    raw = resp.get('raw_api_response') or {}
    if err is None and isinstance(raw, dict):
        err = raw.get('error')
    choices = raw.get('choices') or [] if isinstance(raw, dict) else []
    if (err is None and choices and isinstance(choices[0], dict)):
        err = choices[0].get('error')
    return err


def parse_model(input_root, model_slug, persona_group, personas, rules,
                out_dir, blank_ground_truth=False, limit=None, progress=True,
                results_filename='results.csv', parser_version='parser_v2',
                read_workers=1):
    """Parse seluruh persona untuk satu model. Kembalikan ringkasan statistik.

    blank_ground_truth=True dipakai uji V2 (no-gold-peek).
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, results_filename)
    fd, temporary_path = tempfile.mkstemp(prefix=f'.{results_filename}.',
                                          suffix='.tmp', dir=out_dir)
    os.close(fd)
    stats = defaultdict(lambda: defaultdict(int))
    n_rows = 0

    with open(temporary_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=schema.COLUMNS, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore')
        w.writeheader()

        for persona in personas:
            folder = os.path.join(input_root, model_slug, persona)
            if not os.path.isdir(folder):
                print(f'  [WARN] tidak ada folder: {folder}')
                continue
            files = sorted(f for f in os.listdir(folder)
                           if f.startswith('gsm8k_') and f.endswith('.json'))
            if limit:
                files = files[:limit]

            paths = [os.path.join(folder, fname) for fname in files]
            if read_workers > 1:
                with ThreadPoolExecutor(max_workers=read_workers) as executor:
                    loaded = list(executor.map(_load_json_payload_safe, paths))
            else:
                loaded = [_load_json_payload_safe(path) for path in paths]

            for fname, load_result in zip(files, loaded):
                fpath = os.path.join(folder, fname)
                data, input_json_valid, json_load_error, load_error = load_result
                if load_error is not None:
                    e = load_error
                    qid = os.path.splitext(fname)[0].replace('gsm8k_', '').lstrip('0') or '0'
                    row = {c: '' for c in schema.COLUMNS}
                    row.update({
                        'model': model_slug, 'persona_group': persona_group,
                        'persona': persona, 'question_id': qid,
                        'file': f'{persona}/{fname}', 'answer_status': 'missing_content',
                        'rule_pack': rules.RULE_PACK, 'abstain_reason': 'invalid_json',
                        'input_json_valid': False, 'technical_failure': True,
                        'content_empty': True, 'error_class': type(e).__name__,
                        'error_message': str(e).replace('\r', ' ').replace('\n', ' ')[:500],
                        'parser_version': parser_version,
                    })
                    w.writerow(row)
                    n_rows += 1
                    s = stats[persona]
                    s['total'] += 1; s['missing_content'] += 1
                    s['ab_invalid_json'] += 1; s['src_none'] += 1
                    continue

                inp = data.get('input') or {}
                resp = data.get('response') or {}
                gt_raw = '' if blank_ground_truth else (inp.get('ground_truth') or '')

                content = get_content(resp)
                fr = _finish_reason(resp)
                success = resp.get('success') is True
                res = extract(content, rules,
                              finish_reason=('error' if not success else fr))
                gold, correct = score(res, gt_raw)

                u = _usage(resp)
                perf = resp.get('performance') or {}
                err = _response_error(resp)
                technical_failure = (not success) or fr == 'error' or bool(err)
                row = {
                    'model': model_slug,
                    'persona_group': persona_group,
                    'persona': persona,
                    'question_id': inp.get('question_id', ''),
                    'file': f'{persona}/{fname}',
                    'ground_truth_raw': gt_raw,
                    'ground_truth_answer': '' if gold is None else gold,
                    'response_content_full': content,
                    'response_success': success,
                    'response_error_present': bool(err),
                    'error_class': (
                        'provider_error' if err else
                        'finish_reason_error' if fr == 'error' else
                        'JSONDecodeError_recovered' if not input_json_valid else ''
                    ),
                    'error_message': (
                        str(err).replace('\r', ' ').replace('\n', ' ')[:500]
                        if err else json_load_error[:500]
                    ),
                    'input_json_valid': input_json_valid,
                    'technical_failure': technical_failure,
                    'content_empty': not bool((content or '').strip()),
                    'reasoning_present': has_separate_reasoning(resp),
                    'think_block_stripped': 'think_block_stripped' in res.flags,
                    'extracted_answer': '' if res.value is None else res.value,
                    'candidate_answer': '' if res.candidate is None else res.candidate,
                    'answer_status': res.status,
                    'rule_pack': rules.RULE_PACK,
                    'parse_source': res.source,
                    'priority_hit': res.priority_hit,
                    'extraction_confidence': res.confidence,
                    'abstain_reason': res.abstain_reason,
                    'is_correct': correct,
                    'finish_reason': fr or '',
                    'truncated_by_metadata': fr == 'length',
                    'truncated_by_content': looks_truncated(content, fr),
                    'prompt_tokens': u['prompt_tokens'],
                    'completion_tokens': u['completion_tokens'],
                    'reasoning_tokens': u['reasoning_tokens'],
                    'total_tokens': u['total_tokens'],
                    'latency_sec': perf.get('latency_sec', ''),
                    'retry_count': perf.get('retry_count', ''),
                    'parser_version': parser_version,
                    'text_tail': (content or '').strip()[-200:].replace('\n', ' '),
                }
                w.writerow(row)
                n_rows += 1

                s = stats[persona]
                s['total'] += 1
                s[res.status] += 1
                if correct:
                    s['correct'] += 1
                if res.abstain_reason:
                    s['ab_' + res.abstain_reason] += 1
                s['src_' + (res.source or 'none')] += 1
                for k in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens'):
                    s['sum_' + k] += u[k]

            if progress:
                s = stats[persona]
                acc = 100.0 * s['correct'] / s['total'] if s['total'] else 0
                cr = 100.0 * s['committed'] / s['total'] if s['total'] else 0
                print(f'  {persona:<24} n={s["total"]:>5}  acc={acc:6.2f}%  commit={cr:6.2f}%')

    os.replace(temporary_path, results_path)
    return {'rows': n_rows, 'stats': stats, 'results_path': results_path}


def _csv_bool(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _csv_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def reparse_results(source_results_path, rules, out_dir,
                    blank_ground_truth=False, results_filename='results.csv',
                    parser_version='parser_v2', raw_root=None,
                    refresh_files=None, read_workers=1):
    """Reparse `response_content_full` dari results.csv tanpa membuka raw JSON.

    Jalur ini setara untuk ekstraksi: kolom tersebut adalah `response.content`
    utuh yang ditulis `parse_model`. Metadata teknis dan provenance lama tetap
    dipertahankan; hanya kolom hasil parser yang dihitung ulang. Ini ditujukan
    untuk loop audit rule pack yang berulang dan tidak melakukan API call.

    Jika `refresh_files` diberikan, baris tersebut disegarkan dahulu dari JSON
    di `raw_root`. Ini dipakai sesudah selective rerun agar tidak perlu membaca
    ulang puluhan ribu JSON hanya untuk beberapa target yang berubah.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, results_filename)
    fd, temporary_path = tempfile.mkstemp(prefix=f'.{results_filename}.',
                                          suffix='.tmp', dir=out_dir)
    os.close(fd)
    stats = defaultdict(lambda: defaultdict(int))
    n_rows = 0
    refresh = {str(path).replace('\\', '/').lstrip('/')
               for path in (refresh_files or [])}
    if refresh and not raw_root:
        raise ValueError('raw_root wajib bila refresh_files diberikan')
    refreshed = set()
    refresh_payloads = {}

    # Selective reruns can still contain thousands of large reasoning payloads.
    # Decode those JSON files in parallel once, then preserve the deterministic
    # order of the source CSV while reparsing/scoring below.
    if refresh:
        ordered_refresh = sorted(refresh)
        raw_paths = [os.path.join(raw_root, *path.split('/'))
                     for path in ordered_refresh]
        if read_workers > 1:
            with ThreadPoolExecutor(max_workers=read_workers) as executor:
                loaded_refresh = list(
                    executor.map(_load_json_payload_safe, raw_paths)
                )
        else:
            loaded_refresh = [_load_json_payload_safe(path) for path in raw_paths]
        for relative_file, raw_path, loaded in zip(
                ordered_refresh, raw_paths, loaded_refresh):
            data, input_json_valid, json_load_error, load_error = loaded
            if load_error is not None:
                raise RuntimeError(
                    f'gagal membaca target refresh {raw_path}: {load_error}'
                ) from load_error
            refresh_payloads[relative_file] = (
                data, input_json_valid, json_load_error
            )

    try:
        with open(source_results_path, encoding='utf-8-sig', newline='') as src, \
                open(temporary_path, 'w', encoding='utf-8', newline='') as dst:
            rd = csv.DictReader(src)
            missing = [c for c in schema.COLUMNS if c not in (rd.fieldnames or [])]
            if missing:
                raise ValueError('results sumber tidak memiliki kolom: ' +
                                 ', '.join(missing))
            w = csv.DictWriter(dst, fieldnames=schema.COLUMNS,
                               quoting=csv.QUOTE_ALL, extrasaction='ignore')
            w.writeheader()

            for row in rd:
                relative_file = (row.get('file') or '').replace('\\', '/').lstrip('/')
                if relative_file in refresh:
                    data, input_json_valid, json_load_error = refresh_payloads[
                        relative_file
                    ]
                    inp = data.get('input') or {}
                    resp = data.get('response') or {}
                    content = get_content(resp)
                    success = resp.get('success') is True
                    fr = _finish_reason(resp)
                    err = _response_error(resp)
                    usage = _usage(resp)
                    perf = resp.get('performance') or {}
                    gt_raw = '' if blank_ground_truth else (inp.get('ground_truth') or '')
                    row.update({
                        'question_id': inp.get('question_id', row.get('question_id', '')),
                        'ground_truth_raw': gt_raw,
                        'response_content_full': content,
                        'response_success': success,
                        'response_error_present': bool(err),
                        'error_class': (
                            'provider_error' if err else
                            'finish_reason_error' if fr == 'error' else
                            'JSONDecodeError_recovered' if not input_json_valid else ''
                        ),
                        'error_message': (
                            json.dumps(err, ensure_ascii=False)[:500]
                            if isinstance(err, (dict, list)) else
                            str(err)[:500] if err else json_load_error[:500]
                        ),
                        'input_json_valid': input_json_valid,
                        'technical_failure': (
                            (not success) or fr == 'error' or bool(err)
                        ),
                        'reasoning_present': has_separate_reasoning(resp),
                        'finish_reason': fr or '',
                        'prompt_tokens': usage['prompt_tokens'],
                        'completion_tokens': usage['completion_tokens'],
                        'reasoning_tokens': usage['reasoning_tokens'],
                        'total_tokens': usage['total_tokens'],
                        'latency_sec': perf.get('latency_sec', ''),
                        'retry_count': perf.get('retry_count', ''),
                    })
                    refreshed.add(relative_file)

                content = row.get('response_content_full') or ''
                success = _csv_bool(row.get('response_success'))
                fr = row.get('finish_reason') or None
                res = extract(content, rules,
                              finish_reason=('error' if not success else fr))
                gt_raw = '' if blank_ground_truth else (row.get('ground_truth_raw') or '')
                gold, correct = score(res, gt_raw)

                row.update({
                    'ground_truth_raw': gt_raw,
                    'ground_truth_answer': '' if gold is None else gold,
                    'content_empty': not bool(content.strip()),
                    'think_block_stripped': 'think_block_stripped' in res.flags,
                    'extracted_answer': '' if res.value is None else res.value,
                    'candidate_answer': '' if res.candidate is None else res.candidate,
                    'answer_status': res.status,
                    'rule_pack': rules.RULE_PACK,
                    'parse_source': res.source,
                    'priority_hit': res.priority_hit,
                    'extraction_confidence': res.confidence,
                    'abstain_reason': res.abstain_reason,
                    'is_correct': correct,
                    'truncated_by_metadata': fr == 'length',
                    'truncated_by_content': looks_truncated(content, fr),
                    'parser_version': parser_version,
                    'text_tail': content.strip()[-200:].replace('\n', ' '),
                })
                w.writerow(row)
                n_rows += 1

                s = stats[row.get('persona') or '']
                s['total'] += 1
                s[res.status] += 1
                if correct:
                    s['correct'] += 1
                if res.abstain_reason:
                    s['ab_' + res.abstain_reason] += 1
                s['src_' + (res.source or 'none')] += 1
                for key in ('prompt_tokens', 'completion_tokens',
                            'reasoning_tokens', 'total_tokens'):
                    s['sum_' + key] += _csv_int(row.get(key))

            missing_refresh = sorted(refresh - refreshed)
            if missing_refresh:
                raise ValueError('target refresh tidak ditemukan di results sumber: ' +
                                 ', '.join(missing_refresh[:10]))

        os.replace(temporary_path, results_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return {'rows': n_rows, 'stats': stats, 'results_path': results_path}


ACC_COLUMNS = [
    'model', 'rule_pack', 'persona', 'total', 'correct', 'accuracy',
    'committed', 'commit_rate', 'abstain', 'missing', 'candidate_only',
    'accuracy_on_committed',
    'avg_prompt_tokens', 'avg_completion_tokens', 'avg_reasoning_tokens', 'avg_total_tokens',
]


def write_accuracy(stats, model_slug, rule_pack, out_dir, filename='accuracy.csv'):
    """Penyebut akurasi = SELURUH unit (Bab IV 1275-1276). Tidak ada yang dibuang."""
    rows, tot = [], defaultdict(int)
    for persona in sorted(stats):
        s = stats[persona]
        n = s['total']
        if not n:
            continue
        rows.append(_acc_row(model_slug, rule_pack, persona, s))
        for k, v in s.items():
            tot[k] += v
    if tot['total']:
        rows.append(_acc_row(model_slug, rule_pack, 'OVERALL', tot))
    path = os.path.join(out_dir, filename)
    _write_csv_atomic(path, ACC_COLUMNS, rows)
    return path, rows


def _acc_row(model_slug, rule_pack, persona, s):
    n = s['total']
    committed = s['committed']
    return {
        'model': model_slug, 'rule_pack': rule_pack, 'persona': persona,
        'total': n, 'correct': s['correct'],
        'accuracy': round(100.0 * s['correct'] / n, 4),
        'committed': committed,
        'commit_rate': round(100.0 * committed / n, 4),
        'abstain': s['abstain'], 'missing': s['missing_content'],
        'candidate_only': s['ab_candidate_only'],
        'accuracy_on_committed': round(100.0 * s['correct'] / committed, 4) if committed else 0.0,
        'avg_prompt_tokens': round(s['sum_prompt_tokens'] / n, 2),
        'avg_completion_tokens': round(s['sum_completion_tokens'] / n, 2),
        'avg_reasoning_tokens': round(s['sum_reasoning_tokens'] / n, 2),
        'avg_total_tokens': round(s['sum_total_tokens'] / n, 2),
    }


COV_COLUMNS = ['model', 'rule_pack', 'persona', 'total', 'committed', 'commit_rate',
               'src_hashes', 'src_boxed', 'src_marker_line', 'src_bold_4a', 'src_currency_line',
               'src_bold_4d', 'src_bold_4e', 'src_conclusion', 'src_final_sentence', 'src_none',
               'ab_candidate_only', 'ab_no_final_answer', 'ab_no_number', 'ab_refusal',
               'ab_truncated', 'ab_unclosed_think', 'ab_empty_content']


def write_coverage(stats, model_slug, rule_pack, out_dir, filename='coverage.csv'):
    rows, tot = [], defaultdict(int)
    for persona in sorted(stats):
        s = stats[persona]
        if not s['total']:
            continue
        rows.append(_cov_row(model_slug, rule_pack, persona, s))
        for k, v in s.items():
            tot[k] += v
    if tot['total']:
        rows.append(_cov_row(model_slug, rule_pack, 'OVERALL', tot))
    path = os.path.join(out_dir, filename)
    _write_csv_atomic(path, COV_COLUMNS, rows)
    return path


def _cov_row(model_slug, rule_pack, persona, s):
    row = {'model': model_slug, 'rule_pack': rule_pack, 'persona': persona,
           'total': s['total'], 'committed': s['committed'],
           'commit_rate': round(100.0 * s['committed'] / s['total'], 4)}
    for c in COV_COLUMNS[6:]:
        row[c] = s[c]
    return row


def split_per_persona(results_path, out_dir):
    """Pecah results.csv jadi results_<persona>.csv."""
    grouped = defaultdict(list)
    with open(results_path, encoding='utf-8') as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames
        for row in rd:
            grouped[row['persona']].append(row)
    for persona, rows in grouped.items():
        _write_csv_atomic(os.path.join(out_dir, f'results_{persona}.csv'),
                          fields, rows, quoting=csv.QUOTE_ALL)
