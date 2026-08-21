"""Draw stratified audit samples so every rule can be checked by hand.

    python "src/MAIN_PARSING/ETHICS/audit_sampler.py" --group gender
    python "src/MAIN_PARSING/ETHICS/audit_sampler.py" --group gender --rule leading_digit
    python "src/MAIN_PARSING/ETHICS/audit_sampler.py" --group gender --abstain

Quotas follow AGENT.md 3.1 (rule of three: n clean samples -> error ceiling
about 3/n):

    explicit rules  50 hits  -> ceiling about 6%
    semantic rules 100 hits  -> ceiling about 3%
    fewer than 50 hits       -> review all of them

Sampling is stratified by model, because a rule that only misfires on one
model would be invisible in a global random sample.

The output CSV deliberately does NOT contain the reference label. The auditor
judges what the MODEL answered, not whether the model was right - otherwise the
audit measures agreement with the benchmark instead of extraction precision.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import GROUPS, PARSE_CACHE_ROOT, REPORT_ROOT  # noqa: E402

EXPLICIT_QUOTA = 50
SEMANTIC_QUOTA = 100
MIN_PER_MODEL = 10

_SEMANTIC_PREFIXES = ("conclusion_", "semantic_")

AUDIT_COLUMNS = (
    "audit_verdict",        # filled in by the auditor: ok / wrong / unclear
    "audit_note",
    "model",
    "persona_folder",
    "question_id",
    "subset",
    "parse_source",
    "extracted_answer",
    "extracted_stance",
    "digit_convention",
    "answer_status",
    "abstain_reason",
    "response_flags",
    "evidence",
    "response_content_full",
)


def _load(group_folder: str) -> list[dict]:
    rows: list[dict] = []
    for jsonl in sorted((PARSE_CACHE_ROOT / group_folder).rglob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _stratified(rows: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """Sample up to `quota`, spread across models, at least MIN_PER_MODEL each."""
    if len(rows) <= quota:
        return list(rows)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    picked: list[dict] = []
    for model in sorted(by_model):
        pool = by_model[model]
        rng.shuffle(pool)
        picked.extend(pool[:min(MIN_PER_MODEL, len(pool))])

    remaining = [r for r in rows if r not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, quota - len(picked))])
    return picked[:max(quota, len(picked))]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(AUDIT_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if k.startswith("audit_") else r.get(k, "")) for k in AUDIT_COLUMNS})


def semantic_sample(size: int, seed: int, only_folder: str | None = None) -> list[dict]:
    """Proportional random sample across every semantic (unaudited) row.

    The per-rule quota in AGENT.md 3.1 would need 1,890 rows read by hand. This
    draws one pooled sample instead, which answers a different but useful
    question: "how precise is the semantic layer as a whole?" - with a
    confidence interval, at a fraction of the reading.

    It does NOT tell you which rule is at fault. If precision comes back low,
    fall back to per-rule sampling for the biggest rules.
    """
    rows: list[dict] = []
    for folder in ([only_folder] if only_folder else list(GROUPS)):
        root = PARSE_CACHE_ROOT / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*.jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("extraction_confidence") == "semantic_unaudited":
                        rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:size]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified audit sampler")
    ap.add_argument("--group", default="gender")
    ap.add_argument("--semantic", action="store_true",
                    help="pooled random sample across all semantic rows, every group")
    ap.add_argument("--size", type=int, default=200, help="sample size for --semantic")
    ap.add_argument("--all-groups", action="store_true",
                    help="pool every persona group instead of just --group")
    ap.add_argument("--rule", default=None, help="only this parse_source / candidate_source")
    ap.add_argument("--abstain", action="store_true", help="sample abstentions instead of commits")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    folder = slug_to_folder.get(args.group)
    if folder is None:
        ap.error(f"--group must be one of {sorted(slug_to_folder)}")

    if args.semantic:
        # Restricted to one persona group by default: the thesis analyses
        # Gender Persona, so auditing that group answers the question that
        # actually matters instead of averaging over groups it does not use.
        sample = semantic_sample(args.size, args.seed,
                                 None if args.all_groups else folder)
        if not sample:
            print("tidak ada baris semantic_unaudited - jalankan promote_rules.py dulu")
            return
        tag = "all" if args.all_groups else args.group
        path = REPORT_ROOT / "_audit" / f"audit_semantic_{tag}_{args.size}.csv"
        _write(path, sample)
        print(f"\n{'=' * 78}\n  SAMPEL AUDIT SEMANTIK (acak, seluruh kelompok)\n{'=' * 78}")
        print(f"  {len(sample)} baris -> {path}")
        by_rule = Counter(r["parse_source"] for r in sample)
        print(f"\n  {'aturan':<38}{'sampel':>8}")
        for rule, n in by_rule.most_common():
            print(f"  {rule:<38}{n:>8}")
        print("\n  Isi kolom audit_verdict: ok / wrong / unclear.")
        print("  Nilai APA JAWABAN MODEL, bukan apakah model benar.")
        print("  Lembar ini sengaja tidak memuat kunci jawaban.\n")
        return

    rows = _load(folder)
    if not rows:
        print("cache kosong - jalankan run_parse.py dulu")
        return

    rng = random.Random(args.seed)
    out_dir = REPORT_ROOT / args.group / "_audit"

    if args.abstain:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["answer_status"] == "abstain":
                buckets[r["abstain_reason"] or "unspecified"].append(r)
        label = "abstain_reason"
    else:
        buckets = defaultdict(list)
        for r in rows:
            if r["answer_status"] == "committed":
                buckets[r["parse_source"]].append(r)
            elif r["candidate_source"]:
                buckets[f"candidate::{r['candidate_source']}"].append(r)
        label = "parse_source"

    if args.rule:
        buckets = {k: v for k, v in buckets.items() if k == args.rule}
        if not buckets:
            print(f"tidak ada baris dengan {label} = {args.rule}")
            return

    print(f"\n{'=' * 78}\n  SAMPEL AUDIT - {folder} ({label})\n{'=' * 78}")
    print(f"  {'rule':<34} {'total hit':>10} {'disampel':>9}  kuota")
    print("  " + "-" * 74)

    for rule in sorted(buckets, key=lambda k: -len(buckets[k])):
        pool = buckets[rule]
        semantic = rule.startswith("candidate::") or rule.startswith(_SEMANTIC_PREFIXES)
        quota = SEMANTIC_QUOTA if semantic else EXPLICIT_QUOTA
        if len(pool) < quota:
            sample = list(pool)
            note = "semua (< kuota)"
        else:
            sample = _stratified(pool, quota, rng)
            note = f"{quota}"
        safe = rule.replace("::", "_").replace("/", "_")
        _write(out_dir / f"audit_{args.group}_{safe}.csv", sample)
        print(f"  {rule:<34} {len(pool):>10} {len(sample):>9}  {note}")

    print(f"\n  Ditulis ke: {out_dir}")
    print("  Isi kolom audit_verdict dengan ok / wrong / unclear.")
    print("  Satu 'wrong' -> perbaiki aturannya lalu sampel ulang DARI NOL,")
    print("  bukan dikurangi dari kuota (AGENT.md 3.1).\n")


if __name__ == "__main__":
    main()
