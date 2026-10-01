"""Read-only semantic replay of existing model/core reports with frozen-v2."""
import csv
import os
from pathlib import Path
import tempfile

from .frozen_v2_skeleton import extract
from ..Religion_Persona.run_frozen_v2 import verify_source_lock
from ..Religion_Persona import rules_core, rules_llama33_70b_frozen_v2

PACKS = {
    'meta-llama_llama-3.3-70b-instruct': rules_llama33_70b_frozen_v2,
}
FIELDS = ['extracted_answer', 'candidate_answer', 'answer_status', 'parse_source',
          'priority_hit', 'extraction_confidence', 'abstain_reason',
          'think_block_stripped']


def _value(result, field):
    if field == 'extracted_answer': return '' if result.value is None else str(result.value)
    if field == 'candidate_answer': return '' if result.candidate is None else str(result.candidate)
    if field == 'answer_status': return result.status
    if field == 'parse_source': return result.source
    if field == 'priority_hit': return result.priority_hit or ''
    if field == 'extraction_confidence': return result.confidence
    if field == 'abstain_reason': return result.abstain_reason
    if field == 'think_block_stripped': return str('think_block_stripped' in result.flags)


def replay(report_root, summary_path):
    verify_source_lock()
    summary = []
    for model, model_pack in PACKS.items():
        for filename, pack in [('results.csv', model_pack), ('results_core.csv', rules_core)]:
            path = Path(report_root) / model / filename
            total = mismatches = 0; by_field = {f: 0 for f in FIELDS}; examples = []
            with path.open(encoding='utf-8-sig', newline='') as fh:
                for row in csv.DictReader(fh):
                    total += 1
                    if row.get('input_json_valid') == 'False':
                        # Invalid JSON rows were created directly by runner;
                        # there is no response text to replay.
                        continue
                    finish = 'error' if row.get('response_success') != 'True' else row.get('finish_reason') or None
                    result = extract(row.get('response_content_full', ''), pack, finish_reason=finish)
                    diff = [f for f in FIELDS if row.get(f, '') != _value(result, f)]
                    if diff:
                        mismatches += 1
                        for f in diff: by_field[f] += 1
                        if len(examples) < 3: examples.append(f"{row['persona']}/{row['question_id']}:{','.join(diff)}")
                    if total % 2000 == 0:
                        print(model, filename, total, mismatches, flush=True)
            summary.append({'model': model, 'file': filename, 'rows': total,
                            'mismatches': mismatches, 'match': mismatches == 0,
                            **{f'mismatch_{k}': v for k, v in by_field.items()},
                            'examples': '; '.join(examples)})
            print(model, filename, total, mismatches, flush=True)
    fields = list(summary[0])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + summary_path.name + '.', suffix='.tmp',
                               dir=summary_path.parent)
    os.close(fd)
    try:
        with open(tmp, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        os.replace(tmp, summary_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return summary


if __name__ == '__main__':
    root = Path(r'result\GMS8K\reports_parser_v2\Religion_Persona')
    out = Path(r'result\GMS8K\audit\frozen_v2_semantic_replay.csv')
    rows = replay(root, out)
    raise SystemExit(0 if all(r['match'] for r in rows) else 1)
