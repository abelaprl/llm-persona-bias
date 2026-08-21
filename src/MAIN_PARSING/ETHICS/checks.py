"""Validation checks that have to pass before the numbers are used.

    python "src/MAIN_PARSING/ETHICS/checks.py" --blind
    python "src/MAIN_PARSING/ETHICS/checks.py" --persona-independence
    python "src/MAIN_PARSING/ETHICS/checks.py" --regression
    python "src/MAIN_PARSING/ETHICS/checks.py" --all

--blind
    Mechanical proof of AGENT.md 2.1: strip the reference label from a sample of
    result files, re-parse, and require every extraction column to come out
    byte-identical. If the extractor ever peeked at the label, this fails.

--persona-independence
    Chi-square on (persona x answer_status) per model. A significant result
    means extraction succeeds at different rates depending on persona, so
    accuracy compared across personas carries differential measurement error.
    This is expected to fire - the point is to quantify and report it, not to
    hide it (AGENT.md 2.3).

--regression
    Compare against result/reports/v9_pilot/. Every row that moved from
    committed to abstain has to be explainable; the old engine harvested digits
    out of the phrase "a 0 or 1", so most of them should be false positives
    that this pipeline correctly drops.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")

from config import GROUPS, PARSE_CACHE_ROOT, RESULT_ROOT  # noqa: E402
from run_parse import parse_one  # noqa: E402

#: Columns produced without ever seeing the reference label.
_EXTRACTION_COLUMNS = (
    "extracted_answer", "extracted_stance", "answer_status", "parse_source",
    "digit_convention", "candidate_answer", "candidate_source",
    "behavior_category", "abstain_reason", "extraction_confidence",
    "numeric_semantic_conflict", "response_flags",
)
#: Columns that legitimately depend on the label.
_LABEL_COLUMNS = ("ground_truth", "gt_stance", "is_correct")


def _load_cache(group_folder: str) -> list[dict]:
    root = PARSE_CACHE_ROOT / group_folder
    rows: list[dict] = []
    for jsonl in sorted(root.rglob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------


def check_blind(group_folder: str, sample: int, seed: int = 20260813) -> int:
    print(f"\n{'=' * 78}\n  UJI BUTA GROUND TRUTH ({sample} file acak)\n{'=' * 78}")
    files = [p for p in (RESULT_ROOT / group_folder).rglob("*.json") if "warmup" not in p.parts]
    if not files:
        print("  [SKIP] tidak ada file")
        return 0

    rng = random.Random(seed)
    picked = rng.sample(files, min(sample, len(files)))

    tmp_dir = Path(__file__).resolve().parent / ".blind_tmp"
    tmp_dir.mkdir(exist_ok=True)
    mismatches = 0
    label_seen = 0

    try:
        for path in picked:
            original = parse_one(path)
            if original is None or "_error" in original:
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            stripped = copy.deepcopy(payload)
            for key in ("GT", "GT_label_text", "ground_truth_label", "prediction_label", "is_correct"):
                stripped.get("analysis", {}).pop(key, None)
            stripped.get("metadata", {}).pop("label", None)

            # Mirror the real directory depth so parse_one reads the same
            # model / persona / group names from the path.
            mirror = tmp_dir / path.parts[-4] / path.parts[-3] / path.parts[-2]
            mirror.mkdir(parents=True, exist_ok=True)
            blind_path = mirror / path.name
            blind_path.write_text(json.dumps(stripped, ensure_ascii=False), encoding="utf-8")

            blind = parse_one(blind_path)
            blind_path.unlink()

            for column in _EXTRACTION_COLUMNS:
                if original.get(column) != blind.get(column):
                    mismatches += 1
                    print(f"  [FAIL] {path.name}: {column} berubah "
                          f"{original.get(column)!r} -> {blind.get(column)!r}")
                    break
            if blind.get("ground_truth") is not None:
                label_seen += 1
    finally:
        for leftover in sorted(tmp_dir.rglob("*"), reverse=True):
            leftover.rmdir() if leftover.is_dir() else leftover.unlink()
        tmp_dir.rmdir()

    if mismatches:
        print(f"\n  GAGAL: {mismatches} file berubah hasilnya tanpa label.")
        print("  Artinya ekstraksi menyentuh ground truth - langgar AGENT.md 2.1.\n")
        return 1
    print(f"  LOLOS: {len(picked)} file, semua kolom ekstraksi identik tanpa label.")
    print(f"  (label memang hilang pada {len(picked) - label_seen}/{len(picked)} salinan buta)\n")
    return 0


# --------------------------------------------------------------------------


def check_persona_independence(group_folder: str) -> int:
    from scipy.stats import chi2_contingency

    print(f"\n{'=' * 78}\n  UJI INDEPENDENSI PERSONA (chi-square)\n{'=' * 78}")
    rows = _load_cache(group_folder)
    if not rows:
        print("  [SKIP] cache kosong - jalankan run_parse.py dulu")
        return 0

    per_model: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        per_model[r["model"]][r["persona_folder"]][r["answer_status"]] += 1

    print(f"\n  {'model':<40} {'chi2':>10} {'dof':>5} {'p':>12}  {'Cramer V':>9}")
    print("  " + "-" * 80)
    flagged = []
    for model in sorted(per_model):
        personas = sorted(per_model[model])
        statuses = ("committed", "abstain", "missing")
        table = [[per_model[model][p][s] for s in statuses] for p in personas]
        table = [row for row in table if sum(row) > 0]
        cols = [i for i in range(len(statuses)) if any(row[i] for row in table)]
        table = [[row[i] for i in cols] for row in table]
        if len(table) < 2 or len(table[0]) < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        n = sum(sum(row) for row in table)
        k = min(len(table), len(table[0])) - 1
        cramer = (chi2 / (n * k)) ** 0.5 if n and k else 0.0
        mark = "  <-- tidak independen" if p < 0.05 else ""
        print(f"  {model:<40} {chi2:>10.1f} {dof:>5} {p:>12.3e}  {cramer:>9.3f}{mark}")
        if p < 0.05:
            flagged.append((model, cramer))

    print()
    if flagged:
        print("  Tingkat keberhasilan ekstraksi BERGANTUNG pada persona untuk "
              f"{len(flagged)} dari {len(per_model)} model.")
        print("  Konsekuensi: akurasi antar-persona dihitung pada subset terseleksi.")
        print("  Wajib dilaporkan berdampingan dengan commit_rate, dan dibahas")
        print("  sebagai kemungkinan temuan (persona mengubah kepatuhan format)")
        print("  ATAU sebagai keterbatasan pengukuran - keduanya, bukan salah satu.\n")
    else:
        print("  Tidak ada bukti ketergantungan pada persona.\n")
    return 0


# --------------------------------------------------------------------------


def check_regression(group_folder: str, sample: int = 25) -> int:
    import pandas as pd

    slug = GROUPS[group_folder]
    old_root = Path(__file__).resolve().parents[3] / "result" / "reports" / "v9_pilot" / slug
    if not old_root.is_dir():
        print(f"\n  [SKIP] report lama tidak ada: {old_root}")
        return 0

    print(f"\n{'=' * 78}\n  UJI REGRESI TERHADAP report lama\n{'=' * 78}")
    rows = _load_cache(group_folder)
    new_by_model: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for r in rows:
        new_by_model[r["model"]][(r["persona_folder"], r["question_id"])] = r

    grand = Counter()
    for model_dir in sorted(old_root.iterdir()):
        if not model_dir.is_dir():
            continue
        csvs = list(model_dir.glob("*.csv"))
        if not csvs or model_dir.name not in new_by_model:
            continue
        old = pd.read_csv(csvs[0])
        new = new_by_model[model_dir.name]

        transitions = Counter()
        lost_sources = Counter()
        for _, o in old.iterrows():
            key = (o["persona_folder"], o["question_id"])
            n = new.get(key)
            if n is None:
                transitions[("(hilang)", "-")] += 1
                continue
            transitions[(o["answer_status"], n["answer_status"])] += 1
            if o["answer_status"] == "committed" and n["answer_status"] == "abstain":
                lost_sources[str(o.get("parse_source"))] += 1

        gained = transitions[("abstain", "committed")]
        lost = transitions[("committed", "abstain")]
        print(f"\n  {model_dir.name}")
        print(f"    abstain -> committed : {gained}")
        print(f"    committed -> abstain : {lost}")
        if lost_sources:
            top = ", ".join(f"{k} {v}" for k, v in lost_sources.most_common(6))
            print(f"    sumber lama yang hilang: {top}")
        grand["gained"] += gained
        grand["lost"] += lost

    print(f"\n  Total: +{grand['gained']} committed, -{grand['lost']} committed")
    print("  Setiap baris committed -> abstain WAJIB diperiksa manual sebelum")
    print("  angka dipakai. Gunakan audit_sampler.py --regression untuk sampelnya.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validation checks for the ETHICS parser")
    ap.add_argument("--group", default="gender")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--persona-independence", action="store_true")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=300, help="files for --blind")
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    folder = slug_to_folder.get(args.group)
    if folder is None:
        ap.error(f"--group must be one of {sorted(slug_to_folder)}")

    status = 0
    if args.blind or args.all:
        status |= check_blind(folder, args.sample)
    if args.persona_independence or args.all:
        status |= check_persona_independence(folder)
    if args.regression or args.all:
        status |= check_regression(folder)
    if not (args.blind or args.persona_independence or args.regression or args.all):
        ap.print_help()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
