import os, re, sys, time
from collections import Counter

ROOT = r'D:\TugasAkhir\result\benchmarks\gsm8k\runs\main\Gender_Persona'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'update'
MAX_RE = re.compile(rb'("max_tokens"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')
TEMP_RE = re.compile(rb'("temperature"\s*:\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))')

def iter_json():
    for d, _, fs in os.walk(ROOT):
        for name in fs:
            if name.lower().endswith('.json'):
                yield os.path.join(d, name)

def main():
    start = time.time()
    total = changed = missing = multi = malformed = 0
    max_values = Counter(); temp_values = Counter(); missing_examples=[]; multi_examples=[]; write_errors=[]
    for path in iter_json():
        total += 1
        try:
            with open(path, 'rb') as f: data = f.read()
        except Exception as e:
            write_errors.append((path, 'read', repr(e))); continue
        mm = list(MAX_RE.finditer(data)); tm = list(TEMP_RE.finditer(data))
        if len(mm) != 1 or len(tm) != 1:
            if len(mm) == 0 or len(tm) == 0:
                missing += 1
                if len(missing_examples) < 5: missing_examples.append((path, len(mm), len(tm)))
            else:
                multi += 1
                if len(multi_examples) < 5: multi_examples.append((path, len(mm), len(tm)))
            continue
        max_values[mm[0].group(2).decode()] += 1
        temp_values[tm[0].group(2).decode()] += 1
        if MODE == 'update':
            new = MAX_RE.sub(rb'\g<1>4096', data)
            new = TEMP_RE.sub(rb'\g<1>0', new)
            if new != data:
                try:
                    with open(path, 'wb') as f: f.write(new)
                    changed += 1
                except Exception as e:
                    write_errors.append((path, 'write', repr(e)))
        if total % 10000 == 0:
            print(f'progress total={total} changed={changed} missing={missing} multi={multi} elapsed={time.time()-start:.1f}s', flush=True)
    print(f'completed mode={MODE} total={total} changed={changed} missing={missing} multi={multi} errors={len(write_errors)} elapsed={time.time()-start:.1f}s', flush=True)
    print('max_tokens_before=', dict(max_values), flush=True)
    print('temperature_before=', dict(temp_values), flush=True)
    if missing_examples: print('missing_examples=', missing_examples, flush=True)
    if multi_examples: print('multi_examples=', multi_examples, flush=True)
    if write_errors: print('error_examples=', write_errors[:5], flush=True)
    if MODE == 'update' and write_errors:
        return 2
    if missing or multi:
        return 3
    return 0

raise SystemExit(main())
