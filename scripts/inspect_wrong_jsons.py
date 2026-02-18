import csv
import json
import os
from pathlib import Path

# Configuration
MODEL = 'qwen_qwen3-next-80b-a3b-thinking'
REPORTS_DIR = Path('result') / 'reports' / MODEL
RAW_DIR = Path('result') / MODEL
OUT_DIR = Path('result') / 'analysis' / 'wrong_jsons' / MODEL
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COUNT = 100


def process_persona(persona_folder: str):
    csv_path = REPORTS_DIR / persona_folder / 'results.csv'
    if not csv_path.exists():
        print(f'  - No CSV for {persona_folder}, skipping')
        return

    out_path = OUT_DIR / f'{persona_folder}.jsonl'
    written = 0
    with csv_path.open('r', encoding='utf-8', newline='') as f, out_path.open('w', encoding='utf-8') as out:
        reader = csv.DictReader(f)
        for row in reader:
            is_correct = (row.get('is_correct') or '').strip()
            if is_correct == 'True':
                continue
            qid_str = (row.get('question_id') or '').strip()
            if not qid_str:
                continue
            try:
                qid = int(qid_str)
            except Exception:
                continue

            fname = f'gsm8k_{qid:05d}.json'
            json_path = RAW_DIR / persona_folder / fname
            if not json_path.exists():
                # fallback: maybe raw jsons are directly under RAW_DIR
                alt = RAW_DIR / fname
                if alt.exists():
                    json_path = alt
                else:
                    out.write(json.dumps({'error': 'json_not_found', 'question_id': qid, 'expected': str(json_path)}, ensure_ascii=False) + '\n')
                    written += 1
                    if written >= OUTPUT_COUNT:
                        break
                    continue

            try:
                with json_path.open('r', encoding='utf-8') as jf:
                    data = json.load(jf)
            except Exception as e:
                out.write(json.dumps({'error': 'json_read_error', 'question_id': qid, 'path': str(json_path), 'exc': str(e)}, ensure_ascii=False) + '\n')
                written += 1
                if written >= OUTPUT_COUNT:
                    break
                continue

            record = {
                'csv': {
                    'question_id': qid,
                    'ground_truth_answer': row.get('ground_truth_answer',''),
                    'extracted_answer': row.get('extracted_answer',''),
                    'is_correct': is_correct
                },
                'json': data
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
            if written >= OUTPUT_COUNT:
                break

    print(f'Wrote {written} wrong entries for {persona_folder} -> {out_path}')


def main():
    if not REPORTS_DIR.exists():
        print('Reports dir not found:', REPORTS_DIR)
        return
    personas = [p.name for p in REPORTS_DIR.iterdir() if p.is_dir()]
    print('Found personas:', personas)
    for persona in personas:
        process_persona(persona)


if __name__ == '__main__':
    main()
