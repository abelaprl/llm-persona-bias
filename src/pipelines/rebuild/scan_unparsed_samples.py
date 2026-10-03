"""Collect raw samples of still-unparsed (reparse engine -> None) cases for one
model in the v9_pilot tree (Path B), to discover answer patterns the engine misses.

Usage (from D:\\TugasAkhir):
    python src/pipelines/rebuild/scan_unparsed_samples.py --model meta-llama_llama-3.1-8b-instruct
    python src/pipelines/rebuild/scan_unparsed_samples.py --model meta-llama_llama-3.1-8b-instruct --n 30 --tail 400
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_ethics_csv_reparse import reparse_ethics_answer  # noqa: E402

BASE = Path("result/benchmarks/ethics/runs/v9_pilot")
GROUPS = ["Gender Persona", "Neutral Persona", "Religion Persona"]

# Rough buckets to characterise WHY a raw text didn't parse.
BUCKETS = [
    ("empty",            re.compile(r"^\s*$")),
    ("has_0or1_loose",   re.compile(r"(?<!\d)[01](?!\d)")),   # contains a 0/1 somewhere
    ("says_acceptable",  re.compile(r"(?i)\bacceptable\b")),
    ("says_unaccept",    re.compile(r"(?i)\bunacceptable\b")),
    ("says_wrong",       re.compile(r"(?i)\bwrong\b")),
    ("says_right",       re.compile(r"(?i)\bright\b")),
    ("says_yes_no",      re.compile(r"(?i)\b(?:yes|no)\b")),
    ("depends_neither",  re.compile(r"(?i)\b(?:it depends|depends on|neither)\b")),
]


def bucketize(t: str) -> str:
    if not t.strip():
        return "empty"
    for name, pat in BUCKETS[1:]:
        if pat.search(t):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=25, help="samples to print")
    ap.add_argument("--tail", type=int, default=350, help="tail chars to show")
    args = ap.parse_args()

    buckets = Counter()
    samples = []
    total = none = 0

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
            orig = str(pred.get("parsed_answer", "") or "").strip()
            total += 1
            if orig in ("0", "1"):
                continue
            raw = str(pred.get("raw", "") or "")
            ans, src = reparse_ethics_answer(raw)
            if ans in ("0", "1"):
                continue
            none += 1
            b = bucketize(raw)
            buckets[b] += 1
            if len(samples) < args.n:
                samples.append((b, d.get("persona"), d.get("id"),
                                d.get("analysis", {}).get("GT"), raw[-args.tail:]))

    print(f"MODEL: {args.model}")
    print(f"total={total}  still_none={none} ({100*none/total:.1f}%)\n")
    print("-- buckets (why unparsed):")
    for b, n in buckets.most_common():
        print(f"   {n:>6}  {b}")
    print(f"\n-- {len(samples)} samples (tail {args.tail} chars):")
    for b, persona, qid, gt, tail in samples:
        print(f"\n[{b}] {persona} | {qid} | GT={gt}")
        print(f"   ...{tail!r}")


if __name__ == "__main__":
    main()
