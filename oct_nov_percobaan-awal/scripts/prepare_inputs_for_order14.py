import os, json

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs', 'mmlu_women_implicit_strong', '00')
DST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs', 'mmlu_women_implicit_strong', '14')
os.makedirs(DST_DIR, exist_ok=True)

MODEL_ID = 'meta-llama/llama-3.3-70b-instruct:free'
API_KEY_ALIAS = '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'
NEW_ORDER = 14

count = 0
for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith('.json'):
        continue
    src_path = os.path.join(SRC_DIR, fname)
    with open(src_path, 'r', encoding='utf-8') as f:
        session = json.load(f)

    # update fields
    # run_id: replace the order suffix if present, else append
    old_run_id = session.get('run_id', '')
    # try to replace '-00-' with '-14-' otherwise append
    if '-00-' in old_run_id:
        new_run_id = old_run_id.replace('-00-', '-14-')
    else:
        new_run_id = old_run_id + '-14'

    session['run_id'] = new_run_id
    session['order'] = NEW_ORDER
    session['model_id'] = MODEL_ID
    session['api_key_alias'] = API_KEY_ALIAS

    dst_path = os.path.join(DST_DIR, fname)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    count += 1

print(f'Prepared {count} session files in {DST_DIR}')
