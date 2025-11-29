import os
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SRC_TASK = 'mmlu_women_implicit_strong'
DST_TASK = 'mmlu_men_implicit_strong'
ORDER = os.environ.get('TARGET_ORDER', '14')

src_dir = BASE / 'inputs' / SRC_TASK / ORDER.zfill(2)
dst_dir = BASE / 'inputs' / DST_TASK / ORDER.zfill(2)

os.makedirs(dst_dir, exist_ok=True)

count = 0
for p in sorted(src_dir.iterdir()):
    if p.suffix != '.json':
        continue
    with open(p, 'r', encoding='utf-8') as f:
        sess = json.load(f)
    # replace persona
    sess['persona'] = 'man_implicit_strong'
    # update run_id and optionally task in run_id to avoid collisions
    run_id = sess.get('run_id')
    if run_id and 'women' in run_id:
        run_id = run_id.replace('women', 'men')
        sess['run_id'] = run_id
    out_path = dst_dir / (p.name.replace('women', 'men') if 'women' in p.name else p.name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sess, f, ensure_ascii=False, indent=2)
    count += 1

print(f'Created {count} inputs in {dst_dir}')
