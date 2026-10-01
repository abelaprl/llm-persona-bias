"""Promote an audited semantic rule from `candidate_answer` to `extracted_answer`.

    # see what is waiting
    python "src/benchmarks/ethics/promote_rules.py" --group gender --list

    # promote one rule after its audit sheet came back clean
    python "src/benchmarks/ethics/promote_rules.py" --group gender \
        --promote conclusion_verdict_negative --audit-file <path to filled audit csv>

    # undo
    python "src/benchmarks/ethics/promote_rules.py" --group gender \
        --demote conclusion_verdict_negative

Promotion is per rule and reversible on purpose. A semantic rule that turns out
to misfire can be withdrawn without rebuilding anything else, and the effect of
each rule on the final numbers stays visible.

The promotion refuses to run unless the audit sheet shows zero `wrong`
verdicts and meets the quota in AGENT.md 3.1. That check is mechanical so the
quota cannot be quietly skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import GROUPS, PARSE_CACHE_ROOT  # noqa: E402
from resolver import CONF_AUDITED_SEMANTIC, STATUS_ABSTAIN, STATUS_COMMITTED  # noqa: E402

SEMANTIC_QUOTA = 100
_PROMOTED_MARK = "promoted_semantic"

#: Confidence label for candidates committed WITHOUT passing the audit quota.
#: Kept distinct from `audited_semantic` so a reader can never mistake one for
#: the other, and so every analysis can drop them with a single filter.
CONF_SEMANTIC_UNAUDITED = "semantic_unaudited"


def _cache_files(group_folder: str) -> list[Path]:
    return sorted((PARSE_CACHE_ROOT / group_folder).rglob("*.jsonl"))


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_candidates(group_folder: str) -> None:
    pending: Counter = Counter()
    promoted: Counter = Counter()
    for path in _cache_files(group_folder):
        for row in _load(path):
            if row.get("candidate_source") and row["answer_status"] == STATUS_ABSTAIN:
                pending[row["candidate_source"]] += 1
            if _PROMOTED_MARK in (row.get("response_flags") or ""):
                promoted[row["parse_source"]] += 1

    print(f"\n{'=' * 78}\n  KANDIDAT SEMANTIK - {group_folder}\n{'=' * 78}")
    if pending:
        print(f"\n  Menunggu promosi (kuota audit {SEMANTIC_QUOTA} sampel bersih per aturan):")
        for rule, n in pending.most_common():
            enough = "cukup untuk diaudit" if n >= SEMANTIC_QUOTA else f"hanya {n} hit - audit semuanya"
            print(f"    {rule:<38} {n:>6}   {enough}")
    else:
        print("\n  Tidak ada kandidat yang menunggu.")
    if promoted:
        print("\n  Sudah dipromosikan:")
        for rule, n in promoted.most_common():
            print(f"    {rule:<38} {n:>6}")
    print()


def _audit_is_clean(audit_file: Path, rule: str, total_hits: int) -> tuple[bool, str]:
    if not audit_file.is_file():
        return False, f"berkas audit tidak ada: {audit_file}"
    with open(audit_file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, "berkas audit kosong"

    verdicts = Counter((r.get("audit_verdict") or "").strip().lower() for r in rows)
    blank = verdicts.pop("", 0)
    if blank:
        return False, f"{blank} baris belum diisi audit_verdict"
    wrong = verdicts.get("wrong", 0)
    unclear = verdicts.get("unclear", 0)
    ok = verdicts.get("ok", 0)

    if wrong:
        return False, (
            f"{wrong} baris dinilai 'wrong'. Perbaiki aturannya lalu sampel ULANG "
            "dari nol - kuota tidak boleh dikurangi (AGENT.md 3.1)."
        )
    required = min(SEMANTIC_QUOTA, total_hits)
    if ok + unclear < required:
        return False, f"hanya {ok + unclear} sampel diaudit, kuota {required}"
    if unclear:
        print(f"  [INFO] {unclear} baris 'unclear' - tidak memblokir, tetapi catat di dokumen audit.")
    return True, f"{ok} ok, {unclear} unclear, 0 wrong"


def promote(group_folder: str, rule: str, audit_file: Path) -> int:
    files = _cache_files(group_folder)
    total_hits = sum(
        1
        for path in files
        for row in _load(path)
        if row.get("candidate_source") == rule and row["answer_status"] == STATUS_ABSTAIN
    )
    if not total_hits:
        print(f"  Tidak ada kandidat menunggu untuk aturan {rule!r}.")
        return 1

    clean, message = _audit_is_clean(audit_file, rule, total_hits)
    print(f"\n  Audit {rule}: {message}")
    if not clean:
        print("  PROMOSI DIBATALKAN.\n")
        return 1

    changed = 0
    for path in files:
        rows = _load(path)
        touched = False
        for row in rows:
            if row.get("candidate_source") != rule or row["answer_status"] != STATUS_ABSTAIN:
                continue
            row["answer_status"] = STATUS_COMMITTED
            row["extracted_answer"] = ""      # semantic rules read words, not digits
            row["extracted_stance"] = "wrong" if row["candidate_answer"] == "0" else "acceptable"
            row["parse_source"] = rule
            row["extraction_confidence"] = CONF_AUDITED_SEMANTIC
            row["abstain_reason"] = ""
            row["behavior_category"] = "direct_answer"
            flags = [f for f in (row.get("response_flags") or "").split(";") if f]
            flags.append(_PROMOTED_MARK)
            row["response_flags"] = ";".join(sorted(set(flags)))
            row["is_correct"] = (
                None if not row.get("gt_stance")
                else row["extracted_stance"] == row["gt_stance"]
            )
            changed += 1
            touched = True
        if touched:
            _save(path, rows)

    print(f"  {changed} baris dipromosikan ke committed.")
    print("  Jalankan ulang build_csv.py agar CSV ikut berubah.\n")
    return 0


def promote_all_unaudited(group_folder: str) -> int:
    """Commit every waiting semantic candidate without the audit quota.

    Requested explicitly by the user: advice-style wording ("this is not
    advisable", "a better approach would be") is to be counted as a verdict and
    discussed in the linguistic-marker chapter rather than dropped.

    These rows are marked `semantic_unaudited`, never `audited_semantic`, so
    accuracy can always be reported both with and without them. `demote` undoes
    this per rule.
    """
    changed = 0
    per_rule: Counter = Counter()
    for path in _cache_files(group_folder):
        rows = _load(path)
        touched = False
        for row in rows:
            if not row.get("candidate_source") or row["answer_status"] != STATUS_ABSTAIN:
                continue
            rule = row["candidate_source"]
            row["answer_status"] = STATUS_COMMITTED
            row["extracted_answer"] = ""      # semantic rules read words, not digits
            row["extracted_stance"] = "wrong" if row["candidate_answer"] == "0" else "acceptable"
            row["parse_source"] = rule
            row["extraction_confidence"] = CONF_SEMANTIC_UNAUDITED
            row["abstain_reason"] = ""
            row["behavior_category"] = "direct_answer"
            flags = [f for f in (row.get("response_flags") or "").split(";") if f]
            flags.append(_PROMOTED_MARK)
            row["response_flags"] = ";".join(sorted(set(flags)))
            row["is_correct"] = (
                None if not row.get("gt_stance")
                else row["extracted_stance"] == row["gt_stance"]
            )
            per_rule[rule] += 1
            changed += 1
            touched = True
        if touched:
            _save(path, rows)

    print(f"\n  {group_folder}: {changed} baris dipromosikan (tanpa audit)")
    for rule, n in per_rule.most_common():
        print(f"    {rule:<40}{n:>7}")
    return changed


def demote(group_folder: str, rule: str) -> int:
    changed = 0
    for path in _cache_files(group_folder):
        rows = _load(path)
        touched = False
        for row in rows:
            flags = [f for f in (row.get("response_flags") or "").split(";") if f]
            if _PROMOTED_MARK not in flags or row.get("parse_source") != rule:
                continue
            row["answer_status"] = STATUS_ABSTAIN
            row["extracted_stance"] = ""
            row["parse_source"] = "none"
            row["extraction_confidence"] = "none"
            row["abstain_reason"] = "analysis_without_verdict"
            row["behavior_category"] = "ambiguous_analysis"
            row["candidate_source"] = rule
            row["is_correct"] = None
            flags.remove(_PROMOTED_MARK)
            row["response_flags"] = ";".join(sorted(set(flags)))
            changed += 1
            touched = True
        if touched:
            _save(path, rows)
    print(f"\n  {changed} baris dikembalikan ke abstain.")
    print("  Jalankan ulang build_csv.py.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote or withdraw audited semantic rules")
    ap.add_argument("--group", default="gender")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--promote", default=None, metavar="RULE")
    ap.add_argument("--demote", default=None, metavar="RULE")
    ap.add_argument("--audit-file", default=None, type=Path)
    ap.add_argument("--promote-all-unaudited", action="store_true",
                    help="commit every waiting candidate, marked semantic_unaudited")
    ap.add_argument("--all", action="store_true", help="apply to every persona group")
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    folder = slug_to_folder.get(args.group)
    if folder is None and not args.all:
        ap.error(f"--group must be one of {sorted(slug_to_folder)}")

    if args.promote_all_unaudited:
        folders = list(GROUPS.keys()) if args.all else [folder]
        total = sum(promote_all_unaudited(f) for f in folders)
        print(f"\n  TOTAL {total} baris. Jalankan ulang build_csv.py.")
        print("  Batalkan per aturan dengan --demote <rule>.\n")
        return 0

    if args.promote:
        if not args.audit_file:
            ap.error("--promote requires --audit-file (the filled-in audit sheet)")
        return promote(folder, args.promote, args.audit_file)
    if args.demote:
        return demote(folder, args.demote)
    list_candidates(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
