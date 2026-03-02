import json
import csv
from pathlib import Path
import statistics


TRUNCATED_PATH = Path(__file__).resolve().parents[0] / "truncated_mismatches.json"
OUT_CSV = Path(__file__).resolve().parents[0] / "truncated_token_correlation.csv"


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def process():
    with open(TRUNCATED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    total_tokens_list = []
    finish_counts = {}

    for item in data.get("truncated", []):
        file_path = item.get("file")
        row = {
            "file": file_path,
            "gt": item.get("gt"),
            "extracted": item.get("extracted"),
            "snippet": item.get("snippet"),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
            "max_tokens": None,
        }

        if not file_path:
            rows.append(row)
            continue

        p = Path(file_path)
        if not p.exists():
            rows.append(row)
            continue

        try:
            with open(p, "r", encoding="utf-8") as rf:
                j = json.load(rf)

            usage = safe_get(j, "response", "raw_api_response", "usage", default={})
            choices = safe_get(j, "response", "raw_api_response", "choices", default=[])
            finish = None
            if choices and isinstance(choices, list) and len(choices) > 0:
                finish = choices[0].get("finish_reason") or choices[0].get("native_finish_reason")

            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

            max_tokens = safe_get(j, "request", "parameters", "max_tokens")

            row.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "finish_reason": finish,
                "max_tokens": max_tokens,
            })

            if isinstance(total_tokens, (int, float)):
                total_tokens_list.append(int(total_tokens))

            if finish:
                finish_counts[finish] = finish_counts.get(finish, 0) + 1

        except Exception:
            pass

        rows.append(row)

    # write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["file","gt","extracted","prompt_tokens","completion_tokens","total_tokens","finish_reason","max_tokens","snippet"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in writer.fieldnames})

    # print summary
    print(f"Processed {len(rows)} truncated entries, wrote {OUT_CSV}")
    print("Finish reason counts:")
    for k,v in finish_counts.items():
        print(f"  {k}: {v}")

    if total_tokens_list:
        print("Total tokens statistics:")
        print(f"  count: {len(total_tokens_list)}")
        print(f"  min: {min(total_tokens_list)}")
        print(f"  median: {int(statistics.median(total_tokens_list))}")
        print(f"  mean: {int(statistics.mean(total_tokens_list))}")
        print(f"  max: {max(total_tokens_list)}")


if __name__ == '__main__':
    process()
