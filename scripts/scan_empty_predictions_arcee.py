import argparse
import json
from collections import defaultdict
from pathlib import Path


def is_empty_prediction(prediction: dict) -> bool:
    raw = str(prediction.get("raw") or "").strip()
    parsed = str(prediction.get("parsed_answer") or "").strip()
    excuse = str(prediction.get("excuse") or "").strip()
    explanation = str(prediction.get("explanation") or "").strip()
    return not (raw or parsed or excuse or explanation)


def looks_like_length_cutoff(response: dict) -> bool:
    raw = response.get("raw_api_response") or {}
    try:
        choices = raw.get("choices") or []
        if not choices:
            return False
        choice0 = choices[0]
        finish_reason = str(choice0.get("finish_reason") or "").lower()
        message = choice0.get("message") or {}
        content = message.get("content")
        return finish_reason == "length" and content is None
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Delete empty-prediction files")
    args = parser.parse_args()

    base = Path("result/ETHICS/arcee-ai_trinity-large-thinking")
    if not base.exists():
        raise SystemExit(f"Base folder not found: {base}")

    empty_files = []
    by_persona = defaultdict(int)

    for path in base.rglob("*.json"):
        if path.parts[-2] == "warmup":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        prediction = data.get("prediction") or {}
        response = data.get("response") or {}

        if is_empty_prediction(prediction) and response.get("success") is True:
            empty_files.append(path)
            by_persona[path.parent.name] += 1

    print(f"EMPTY_PREDICTION_TOTAL={len(empty_files)}")
    for persona, count in sorted(by_persona.items()):
        print(f"{persona}: {count}")

    out_list = Path("tmp_arcee_empty_predictions.txt")
    out_list.write_text("\n".join([str(p) for p in empty_files]), encoding="utf-8")
    print(f"Saved list to {out_list}")

    if args.delete and empty_files:
        for path in empty_files:
            path.unlink(missing_ok=True)
        print(f"DELETED={len(empty_files)}")

    # Optional: report length-cutoff subset count
    length_cutoff = 0
    for path in empty_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if looks_like_length_cutoff(data.get("response") or {}):
            length_cutoff += 1
    print(f"LENGTH_CUTOFF_LIKELY={length_cutoff}")


if __name__ == "__main__":
    main()
