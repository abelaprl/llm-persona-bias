"""Targeted re-run of the 675 DeepSeek R1 ETHICS v9_pilot 'infra-failure' cases.

These are the cases where the original response was EMPTY or TRUNCATED
(finish_reason == length / error / None) — see
docs/ETHICS_v9_deepseek_parse_failure_diagnosis.md.

Strategy (safest, fully reproducible):
  - For each failed JSON, RESEND THE EXACT SAME request.messages (prompt + warmup
    context identical) with a LARGER max_tokens, so the only change vs the original
    run is the token budget. No prompt/warmup re-derivation.
  - Original files are backed up to result/ETHICS/_rerun_backup/v9_infra_deepseek/
    before being overwritten.
  - Already-fixed files (raw now yields a 0/1) are skipped, so the script is
    resumable / idempotent.

Usage (from D:\\TugasAkhir):
    python src/rerun/rerun_v9_infra_deepseek.py                 # full run
    python src/rerun/rerun_v9_infra_deepseek.py --limit 5       # smoke test
    python src/rerun/rerun_v9_infra_deepseek.py --workers 5
After it finishes, rebuild the CSVs:
    python src/rebuild/build_v9_csv_per_group.py --model deepseek_deepseek-r1-distill-qwen-32b
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path("src/ethics_runner").resolve()))
sys.path.insert(0, str(Path("src/rebuild").resolve()))

from run_ethics_prompt_v9 import (  # noqa: E402
    call_openrouter, clean_api_response, extract_raw_text,
    parse_v9_response, normalize_binary_label, save_json,
    VARIANT_NAME, OUTPUT_NAME,
)
from rebuild_ethics_csv_reparse import reparse_ethics_answer  # noqa: E402

MODEL_SLUG = "deepseek_deepseek-r1-distill-qwen-32b"
MODEL_ID = "deepseek/deepseek-r1-distill-qwen-32b"
LIST_FILE = Path(f"result/ETHICS/{MODEL_SLUG}/v9_infra_rerun_files.txt")
BACKUP_DIR = Path("result/ETHICS/_rerun_backup/v9_infra_deepseek")
RESULT_ROOT = Path("result/ETHICS/v9_pilot")

_print_lock = threading.Lock()
_counter = {"done": 0, "fixed": 0, "still_empty": 0, "skipped": 0, "failed": 0}


def load_model_cfg() -> dict:
    models = json.load(open("config/model_keys.json", encoding="utf-8"))
    for m in models:
        if m.get("model_id") == MODEL_ID:
            return m
    raise RuntimeError(f"{MODEL_ID} not found in config/model_keys.json")


def already_fixed(d: dict) -> bool:
    pred = d.get("prediction", {}) or {}
    if str(pred.get("parsed_answer", "") or "").strip() in ("0", "1"):
        return True
    ans, _ = reparse_ethics_answer(pred.get("raw", ""))
    return ans in ("0", "1")


def process_one(fp: str, cfg: dict, max_tokens: int, total: int) -> None:
    p = Path(fp)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        with _print_lock:
            _counter["failed"] += 1
            print(f"  [READ-FAIL] {p.name}: {e}", flush=True)
        return

    if already_fixed(d):
        with _print_lock:
            _counter["skipped"] += 1; _counter["done"] += 1
        return

    # backup once (don't overwrite an existing backup)
    try:
        rel = p.resolve().relative_to(RESULT_ROOT.resolve())
    except ValueError:
        rel = Path(p.name)
    bpath = BACKUP_DIR / rel
    if not bpath.exists():
        bpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, bpath)

    messages = d["request"]["messages"]
    api = call_openrouter(cfg, messages, temperature=0.0, max_tokens=max_tokens,
                          max_retries=8, retry_base_sec=8.0)
    raw_api = clean_api_response(api["api_response"]) if api["success"] else None
    raw_text = extract_raw_text(raw_api) if raw_api else ""
    explanation, parsed_answer, parsing_failed = parse_v9_response(raw_text)

    gt = d.get("analysis", {}).get("ground_truth_label")
    pred_label = normalize_binary_label(parsed_answer)
    is_correct = (pred_label == gt) if (pred_label is not None and gt is not None) else None

    usage = (raw_api or {}).get("usage", {}) if raw_api else {}
    comp_details = usage.get("completion_tokens_details", {}) or {}

    # update parameters + prediction + response + metadata in place (schema preserved)
    d["request"].setdefault("parameters", {})
    d["request"]["parameters"]["max_tokens"] = max_tokens
    d["request"]["parameters"]["model"] = MODEL_ID
    d["prediction"] = {
        "raw": raw_text,
        "explanation": explanation,
        "parsed_answer": parsed_answer,
        "parsing_failed": parsing_failed,
    }
    d["response"] = {
        "success": api["success"],
        "raw_api_response": raw_api,
        "error": api["error"],
        "retry_count": api["retry_count"],
        "latency_sec": api["latency"],
    }
    meta = d.setdefault("metadata", {})
    meta["prompt_variant"] = VARIANT_NAME
    meta["output_name"] = OUTPUT_NAME
    meta["rerun"] = "v9_infra_maxtokens"
    meta["rerun_timestamp"] = datetime.now().isoformat()
    meta["max_tokens"] = max_tokens
    meta["completion_tokens"] = usage.get("completion_tokens")
    meta["reasoning_tokens"] = comp_details.get("reasoning_tokens")
    d.setdefault("analysis", {})
    d["analysis"]["prediction_label"] = pred_label
    d["analysis"]["is_correct"] = is_correct

    save_json(p, d)

    # bookkeeping
    recovered = parsed_answer in ("0", "1") or reparse_ethics_answer(raw_text)[0] in ("0", "1")
    with _print_lock:
        _counter["done"] += 1
        if recovered:
            _counter["fixed"] += 1
        elif not raw_text.strip():
            _counter["still_empty"] += 1
        if _counter["done"] % 25 == 0 or _counter["done"] == total:
            c = _counter
            print(f"  [{c['done']}/{total}] fixed={c['fixed']} still_empty={c['still_empty']} "
                  f"skipped={c['skipped']} failed={c['failed']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N (smoke test)")
    args = ap.parse_args()

    cfg = load_model_cfg()
    files = [l.strip() for l in LIST_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        files = files[: args.limit]
    total = len(files)
    print("=" * 60)
    print(f"Re-run DeepSeek R1 infra cases: {total} files")
    print(f"max_tokens={args.max_tokens} workers={args.workers}")
    print(f"backup -> {BACKUP_DIR}")
    print("=" * 60, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, fp, cfg, args.max_tokens, total) for fp in files]
        for fut in as_completed(futs):
            exc = fut.exception()
            if exc:
                with _print_lock:
                    print(f"  [THREAD-ERR] {type(exc).__name__}: {exc}", flush=True)

    c = _counter
    print("=" * 60)
    print(f"DONE. processed={c['done']}/{total} fixed={c['fixed']} "
          f"still_empty={c['still_empty']} skipped={c['skipped']} failed={c['failed']}")
    print("Next: python src/rebuild/build_v9_csv_per_group.py --model " + MODEL_SLUG)


if __name__ == "__main__":
    main()
