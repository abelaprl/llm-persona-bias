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


def is_403_error(response: dict) -> bool:
    success = response.get("success")
    err = response.get("error")
    raw = response.get("raw_api_response")

    err_text = ""
    if isinstance(err, str):
        err_text = err
    elif err is not None:
        err_text = str(err)

    raw_text = ""
    if isinstance(raw, dict):
        try:
            raw_text = json.dumps(raw)
        except Exception:
            raw_text = ""

    if success is False:
        if any(k in err_text for k in ["403", "Key limit", "key limit", "HTTP 403"]):
            return True
        if any(k in raw_text for k in ["403", "Key limit", "key limit"]):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_403", action="store_true")
    parser.add_argument("--delete_empty", action="store_true")
    args = parser.parse_args()

    base = Path("result/ETHICS/moonshotai_kimi-k2-thinking")
    if not base.exists():
        raise SystemExit(f"Base folder not found: {base}")

    by_persona_403 = defaultdict(int)
    by_persona_empty = defaultdict(int)
    error_403_files = []
    empty_files = []

    for path in base.rglob("*.json"):
        if path.parts[-2] == "warmup":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        prediction = data.get("prediction") or {}
        response = data.get("response") or {}

        if is_403_error(response):
            error_403_files.append(path)
            by_persona_403[path.parent.name] += 1

        if is_empty_prediction(prediction) and response.get("success") is True:
            empty_files.append(path)
            by_persona_empty[path.parent.name] += 1

    print(f"ERROR_403_TOTAL={len(error_403_files)}")
    for persona, count in sorted(by_persona_403.items()):
        print(f"403 {persona}: {count}")

    print(f"EMPTY_PREDICTION_TOTAL={len(empty_files)}")
    for persona, count in sorted(by_persona_empty.items()):
        print(f"empty {persona}: {count}")

    Path("tmp_moonshot_403_files.txt").write_text(
        "\n".join([str(p) for p in error_403_files]), encoding="utf-8"
    )
    Path("tmp_moonshot_empty_predictions.txt").write_text(
        "\n".join([str(p) for p in empty_files]), encoding="utf-8"
    )

    if args.delete_403 and error_403_files:
        for path in error_403_files:
            path.unlink(missing_ok=True)
        print(f"DELETED_403={len(error_403_files)}")

    if args.delete_empty and empty_files:
        for path in empty_files:
            path.unlink(missing_ok=True)
        print(f"DELETED_EMPTY={len(empty_files)}")


if __name__ == "__main__":
    main()
