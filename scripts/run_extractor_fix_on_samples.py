#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import extractor_fix

ROOT = Path(__file__).resolve().parent.parent
ALL_MISMATCHES = ROOT / 'scripts' / 'all_mismatches_full_responses.json'
OUT_RESULTS = ROOT / 'scripts' / 'test_extractor_fix_src_results.json'
OUT_MISMATCHES = ROOT / 'scripts' / 'test_extractor_fix_src_mismatches.json'

def persona_from_path(fp: str) -> str:
    # try to locate 'result/openai_gpt-oss-120b/' then take next segment as persona
    key = 'result/openai_gpt-oss-120b/'
    if key in fp:
        rest = fp.split(key, 1)[1]
        return rest.split('/')[0].replace('\\', '/')
    # fallback to parent directory name
    return Path(fp).parent.name


def main():
    data = json.loads(ALL_MISMATCHES.read_text(encoding='utf-8'))
    mismatches = data.get('mismatches', [])
    by_persona = defaultdict(list)
    for e in mismatches:
        p = persona_from_path(e.get('file', ''))
        by_persona[p].append(e)

    results = {}
    all_mismatch_out = []
    total_checked = 0
    total_correct = 0
    for persona, items in by_persona.items():
        sel = items[:100]
        persona_ok = 0
        persona_total = len(sel)
        persona_mismatches = []
        for it in sel:
            resp = it.get('response_content')
            gt = it.get('gt')
            extracted = extractor_fix.extract_answer(resp)
            ok = extractor_fix.is_correct(extracted, gt)
            if not ok:
                persona_mismatches.append({
                    'file': it.get('file'),
                    'gt': gt,
                    'extracted': extracted,
                    'response_content': resp,
                })
            else:
                persona_ok += 1
        results[persona] = {'checked': persona_total, 'correct': persona_ok, 'mismatches': len(persona_mismatches)}
        all_mismatch_out.extend(persona_mismatches)
        total_checked += persona_total
        total_correct += persona_ok

    summary = {
        'total_personas': len(results),
        'total_checked': total_checked,
        'total_correct': total_correct,
        'accuracy': total_correct / total_checked if total_checked else None,
        'per_persona': results,
    }
    OUT_RESULTS.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MISMATCHES.write_text(json.dumps({'mismatches': all_mismatch_out}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote', OUT_RESULTS, 'and', OUT_MISMATCHES)
    print('Accuracy:', summary['accuracy'])


if __name__ == '__main__':
    main()
