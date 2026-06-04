import os, json, csv

input_root = r'result/GMS8K/Religion_Persona/deepseek_deepseek-r1-distill-qwen-32b'

empty_files = []
with open(r'result/reports/GSM8K/_tmp_rebuild_results_csv_ministrail/results.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ec = row.get('error_category', '') or ''
        content = row.get('response_content_full', '') or ''
        if ec.strip() == 'no_number_found' and not content.strip():
            empty_files.append(row['file'])

print(f'Empty-content failures: {len(empty_files)}')

content_type_counts = {}
sample_by_type = {}

for fp in empty_files:
    fp_clean = fp.replace('\\\\', os.sep).lstrip(os.sep)
    full = os.path.join(input_root, fp_clean)
    try:
        with open(full, encoding='utf-8') as jf:
            data = json.load(jf)
    except Exception as e:
        print(f'READ ERROR: {full} -- {e}')
        continue

    resp = data.get('response', {}) or {}
    raw_content = resp.get('content', None)
    success = resp.get('success', None)
    error = resp.get('error', None)
    raw_api = resp.get('raw_api_response', None)

    # Classify what's in the content field
    if raw_content is None:
        key = 'content_is_None'
    elif isinstance(raw_content, str) and not raw_content.strip():
        key = 'content_is_empty_string'
    elif isinstance(raw_content, str) and raw_content.strip():
        key = f'content_is_nonempty_string (success={success})'
    elif isinstance(raw_content, list):
        key = f'content_is_list len={len(raw_content)}'
    else:
        key = f'content_is_{type(raw_content).__name__}'

    content_type_counts[key] = content_type_counts.get(key, 0) + 1
    if key not in sample_by_type:
        sample_by_type[key] = {
            'file': fp_clean,
            'success': success,
            'error': str(error)[:200] if error else None,
            'raw_content': repr(str(raw_content)[:300]) if raw_content is not None else None,
            'raw_api_type': type(raw_api).__name__,
            'raw_api_keys': list(raw_api.keys()) if isinstance(raw_api, dict) else None,
        }

print()
print('Content field classification:')
for k, v in sorted(content_type_counts.items(), key=lambda x: -x[1]):
    print(f'  {v:5d}  {k}')

print()
print('Samples per type:')
for k, s in sample_by_type.items():
    print(f'\n  [{k}]')
    for sk, sv in s.items():
        print(f'    {sk}: {sv}')
