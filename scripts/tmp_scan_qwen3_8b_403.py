import json
from pathlib import Path

base = Path("result/ETHICS/qwen_qwen3-8b")
error_files = []

for path in base.rglob("*.json"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        error_files.append((str(path), "invalid_json"))
        continue

    response = data.get("response") or {}
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

    is_403 = False
    if success is False:
        if any(k in err_text for k in ["403", "Key limit", "key limit", "HTTP 403"]):
            is_403 = True
        elif any(k in raw_text for k in ["403", "Key limit", "key limit"]):
            is_403 = True

    if is_403:
        error_files.append((str(path), err_text))

print(f"FOUND_403_ERRORS={len(error_files)}")
for path, err in error_files[:50]:
    print(path)
    if err:
        print(f"  error: {err[:200]}")

out_path = Path("tmp_qwen3_8b_403_errors.txt")
out_path.write_text("\n".join([p for p, _ in error_files]), encoding="utf-8")
print(f"Saved list to {out_path}")
