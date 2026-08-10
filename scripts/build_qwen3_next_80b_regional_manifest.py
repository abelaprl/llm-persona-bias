"""Build exact Regional Cultural Qwen3 Next 80B infrastructure manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import build_qwen3_next_80b_nonregional_manifest as common


OUTPUT_DIR = (
    common.REPO_ROOT / "result" / "rerun_manifests" / "qwen3_next_80b_regional"
)
ETHICS_GROUP = "Regional Cultural Persona"
GSM8K_GROUP = "regional_cultural_persona"
ETHICS_MODEL_DIR = common.ETHICS_ROOT / ETHICS_GROUP / common.MODEL_SLUG
GSM8K_MODEL_DIR = common.GSM8K_ROOT / GSM8K_GROUP / common.MODEL_SLUG
EXPECTED_PERSONAS = {
    "east_asian_explicit",
    "east_asian_implicit_1",
    "east_asian_implicit_2",
    "east_asian_implicit_3",
    "north_american_explicit",
    "north_american_implicit_1",
    "north_american_implicit_2",
    "north_american_implicit_3",
    "south_asian_explicit",
    "south_asian_implicit_1",
    "south_asian_implicit_2",
    "south_asian_implicit_3",
    "sub_saharan_african_explicit",
    "sub_saharan_african_implicit_1",
    "sub_saharan_african_implicit_2",
    "sub_saharan_african_implicit_3",
}


def collect_files() -> tuple[list[common.FileMeta], list[dict[str, object]]]:
    files: list[common.FileMeta] = []
    inventory: list[dict[str, object]] = []
    for dataset, root, group, model_dir, expected in (
        (
            "ETHICS",
            common.ETHICS_ROOT,
            ETHICS_GROUP,
            ETHICS_MODEL_DIR,
            1514,
        ),
        (
            "GSM8K",
            common.GSM8K_ROOT,
            GSM8K_GROUP,
            GSM8K_MODEL_DIR,
            1319,
        ),
    ):
        if not model_dir.is_dir():
            raise RuntimeError(f"Missing Regional Cultural model directory: {model_dir}")
        persona_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
        names = {path.name for path in persona_dirs}
        if names != EXPECTED_PERSONAS:
            raise RuntimeError(
                f"Unexpected {dataset} regional persona set: "
                f"missing={sorted(EXPECTED_PERSONAS - names)}, "
                f"extra={sorted(names - EXPECTED_PERSONAS)}"
            )
        for persona_dir in persona_dirs:
            main_paths = sorted(persona_dir.glob("*.json"))
            if len(main_paths) != expected:
                raise RuntimeError(
                    f"{dataset}/{persona_dir.name} expected {expected} JSON, "
                    f"found {len(main_paths)}"
                )
            inventory.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "persona": persona_dir.name,
                    "count": len(main_paths),
                    "expected": expected,
                }
            )
            for path in main_paths:
                relative = PurePosixPath(path.relative_to(root).as_posix())
                files.append(
                    common.FileMeta(dataset, group, persona_dir.name, path, relative)
                )
    return files, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--expected-ethics", type=int, default=5085)
    parser.add_argument("--expected-gsm8k", type=int, default=546)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    files, inventory = collect_files()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspections = list(executor.map(common.inspect_file, files))
    rows, missing_rows = common.build_rows(inspections, inventory)
    ethics_rows = [row for row in rows if row["dataset"] == "ETHICS"]
    gsm8k_rows = [row for row in rows if row["dataset"] == "GSM8K"]
    if args.expected_ethics and len(ethics_rows) != args.expected_ethics:
        raise RuntimeError(
            f"ETHICS guard expected {args.expected_ethics}, found {len(ethics_rows)}"
        )
    if args.expected_gsm8k and len(gsm8k_rows) != args.expected_gsm8k:
        raise RuntimeError(
            f"GSM8K guard expected {args.expected_gsm8k}, found {len(gsm8k_rows)}"
        )

    common.write_manifest(OUTPUT_DIR / "qwen_regional_ethics_infra.csv", ethics_rows)
    common.write_manifest(OUTPUT_DIR / "qwen_regional_gsm8k_infra.csv", gsm8k_rows)
    common.write_manifest(OUTPUT_DIR / "qwen_regional_infra_all.csv", rows)
    print(f"Scanned main JSON : {len(files)}")
    print(f"ETHICS manifest   : {len(ethics_rows)}")
    print(f"GSM8K manifest    : {len(gsm8k_rows)}")
    print(f"Missing JSON rows : {len(missing_rows)}")
    print(f"Combined manifest : {len(rows)}")
    print(
        "Issue categories  : "
        f"{dict(sorted(Counter(str(row['issue_category']) for row in rows).items()))}"
    )
    print(f"Output directory  : {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
