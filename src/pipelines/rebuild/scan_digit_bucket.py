"""Inspect still-none cases that DO contain a lone 0/1 digit, for one model.
Helps decide whether a new digit-capture pattern is worth adding.

Usage:
    python src/pipelines/rebuild/scan_digit_bucket.py --model meta-llama_llama-3.1-8b-instruct --n 30
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_ethics_csv_reparse import reparse_ethics_answer  # noqa: E402

BASE = Path("result/benchmarks/ethics/runs/v9_pilot")
GROUPS = ["Gender Persona", "Neutral Persona", "Religion Persona"]
DIGIT = re.compile(r"(?<!\d)[01](?!\d)")
# contraction-aware verdict probes (just for counting potential)
CONTRACTION_WRONG = re.compile(r"(?i)\b(?:it|that|this|which|he|she|they)\s*'s\s+(?:\w+\s+){0,3}wrong\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tail", type=int, default=240)
    args = ap.parse_args()

    n_digit = 0
    n_contraction = 0
    samples = []
    for group in GROUPS:
        gdir = BASE / group / args.model
        if not gdir.exists():
            continue
        for jf in gdir.rglob("*.json"):
            if "warmup" in jf.parts:
                continue
            try:
                d = json.load(open(jf, encoding="utf-8"))
            except Exception:
                continue
            pred = d.get("prediction", {}) or {}
            if str(pred.get("parsed_answer", "") or "").strip() in ("0", "1"):
                continue
            raw = str(pred.get("raw", "") or "")
            ans, _ = reparse_ethics_answer(raw)
            if ans in ("0", "1"):
                continue
            if CONTRACTION_WRONG.search(raw):
                n_contraction += 1
            if DIGIT.search(raw):
                n_digit += 1
                if len(samples) < args.n:
                    # show context around the LAST lone digit
                    matches = list(DIGIT.finditer(raw))
                    m = matches[-1]
                    a, b = max(0, m.start() - 90), min(len(raw), m.end() + 40)
                    samples.append((d.get("id"), d.get("analysis", {}).get("GT"),
                                    raw[a:b].replace("\n", " ")))
    print(f"MODEL: {args.model}")
    print(f"still-none containing lone 0/1 digit: {n_digit}")
    print(f"still-none matching contraction-wrong probe: {n_contraction}\n")
    print(f"-- {len(samples)} samples (context around last lone digit):")
    for qid, gt, ctx in samples:
        print(f"\n[{qid} GT={gt}]")
        print(f"   ...{ctx!r}...")


if __name__ == "__main__":
    main()
