import pandas as pd
import json
from pathlib import Path

pairs = [
    (Path('data/SQuAD_train.parquet'), Path('data/SQuAD_train.json')),
    (Path('data/SQuAD_validation.parquet'), Path('data/SQuAD_validation.json')),
]

def to_py(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

for parquet_path, json_path in pairs:
    df = pd.read_parquet(parquet_path)
    records = []
    for row in df.itertuples(index=False):
        rec = {
            "id": row.id,
            "title": row.title,
            "context": row.context,
            "question": row.question,
            "answers": to_py(row.answers),
        }
        records.append(rec)
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f"Wrote {json_path} with {len(records)} items")
