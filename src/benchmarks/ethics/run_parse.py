"""Parse one persona folder at a time into a JSONL cache.

Usage (from D:\\TugasAkhir):

    python "src/benchmarks/ethics/run_parse.py" --group gender
    python "src/benchmarks/ethics/run_parse.py" --group gender --model openai_gpt-oss-20b
    python "src/benchmarks/ethics/run_parse.py" --group gender --persona woman_implicit_1
    python "src/benchmarks/ethics/run_parse.py" --all

Output:
    result/parsing/v9_main/<Group Persona>/<model>/<persona>.jsonl

Splitting parse from CSV assembly is deliberate: reading 72,671 JSON files
takes minutes, so trying a new rule should not mean re-walking the tree.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(errors="replace")  # Windows console vs. model unicode

import detectors  # noqa: E402
import masking  # noqa: E402
import normalize  # noqa: E402
import resolver  # noqa: E402
import scoring  # noqa: E402
from config import (  # noqa: E402
    GROUPS,
    PARSE_CACHE_ROOT,
    RESULT_ROOT,
    TEXT_TAIL_CHARS,
)

#: Persona name to assume when a model folder holds result files directly,
#: with no persona subdirectory. Only the neutral condition was written that
#: way, and only for one model.
DEFAULT_PERSONA_BY_GROUP = {"Neutral Persona": "neutral"}


def _api_bits(payload: dict) -> tuple[str, str]:
    """(finish_reason, reasoning_text) from the stored API response."""
    response = payload.get("response") or {}
    api = response.get("raw_api_response") or {}
    choices = api.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    return (
        str(first.get("finish_reason") or ""),
        str(message.get("reasoning") or "") if isinstance(message, dict) else "",
    )


def _load_result(path: Path) -> tuple[dict | None, str]:
    """Read one result file, recovering from a partially overwritten one.

    A few files contain a complete JSON object followed by leftover bytes from
    a longer earlier write (a rerun wrote a shorter payload without truncating
    the file). The leading object is intact and is the current response, so it
    is used - with a flag, never silently.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"unreadable ({exc})"

    try:
        return json.loads(text), ""
    except ValueError:
        pass

    try:
        payload, _end = json.JSONDecoder().raw_decode(text)
    except ValueError as exc:
        return None, f"invalid json ({exc})"

    if not isinstance(payload, dict):
        return None, "leading value is not an object"
    return payload, "trailing_bytes_recovered"


def parse_one(path: Path, model_name: str = "", group_folder: str = "",
              persona_name: str = "") -> dict | None:
    """Parse a single result file into one output record."""
    payload, load_note = _load_result(path)
    if payload is None:
        return {"_error": f"{load_note}: {path.name}", "file": path.name}
    if not isinstance(payload, dict):
        return {"_error": f"not an object: {path.name}", "file": path.name}

    # Identity comes from the caller, which knows the folder it is walking.
    # The result tree is not uniform - most persona results sit at
    # <group>/<model>/<persona>/, but deepseek's Neutral results were written
    # flat at <group>/<model>/ - so deriving it from path offsets here would
    # silently mislabel one model. The payload's own `persona` field wins when
    # present, since the runner wrote it.
    persona = str(payload.get("persona") or "") or persona_name
    model = model_name
    group = group_folder

    subset = str(payload.get("subset") or "")
    raw = str((payload.get("prediction") or {}).get("raw") or "")
    meta = payload.get("metadata") or {}
    analysis = payload.get("analysis") or {}
    finish_reason, reasoning = _api_bits(payload)

    # ---- extraction: no reference label is visible from here on ----------
    working, pre_flags = normalize.normalise(raw)
    has_text = bool(working.strip())
    masked, mask_flags = masking.mask(working)
    signals = detectors.run_all(working, masked, subset) if has_text else []
    decision = resolver.resolve(signals, working, pre_flags, mask_flags, has_text)

    # An empty content channel while the reasoning channel has text means the
    # model ran out of tokens mid-thought. That is a missing response, not an
    # answer: using the reasoning channel would judge thinking models by a
    # different standard than the rest (AGENT.md 2.2).
    if not has_text and reasoning.strip():
        decision.flags.append("empty_content_with_reasoning")
    if load_note:
        decision.flags.append(load_note)

    # ---- scoring: the label enters only here -----------------------------
    label = analysis.get("ground_truth_label")
    reference = scoring.gt_stance(subset, label)
    correct = scoring.is_correct(decision.extracted_stance, reference)

    prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")
    total = None
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total = prompt_tokens + completion_tokens

    return {
        "model": model,
        "persona_folder": persona,
        "persona_group": GROUPS.get(group, group),
        "file": path.name,
        "question_id": str(payload.get("id") or path.stem),
        "dataset": subset,
        "subset": subset,
        "ground_truth": label,
        "gt_stance": reference,
        "response_content_full": raw,
        "extracted_answer": decision.extracted_answer,
        "extracted_stance": decision.extracted_stance,
        "answer_status": decision.answer_status,
        "parse_source": decision.parse_source,
        "is_correct": correct,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": meta.get("reasoning_tokens"),
        "total_tokens": total,
        "truncated": finish_reason == "length",
        "text_tail": normalize.tail(raw, TEXT_TAIL_CHARS),
        "digit_convention": decision.digit_convention,
        "candidate_answer": decision.candidate_answer,
        "candidate_source": decision.candidate_source,
        "behavior_category": decision.behavior_category,
        "abstain_reason": decision.abstain_reason,
        "extraction_confidence": decision.extraction_confidence,
        "numeric_semantic_conflict": decision.numeric_semantic_conflict,
        "response_flags": ";".join(sorted(set(decision.flags))),
        "construct_valid": scoring.construct_valid(subset),
        "evidence": decision.evidence,
    }


class WorkerPool:
    """A process pool that rebuilds itself after a worker dies.

    Over a full run this maps ~372k files. A single BrokenProcessPool used to
    poison every later persona, because a broken executor rejects all further
    work - the run finished, but serially, taking hours. Recreating the pool
    costs one respawn and keeps the remaining personas parallel.
    """

    def __init__(self, workers: int):
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None
        self.restarts = 0

    def _ensure(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self._pool

    def map(self, fn, items, chunksize: int = 64) -> list:
        if self.workers <= 1:
            return [fn(item) for item in items]
        try:
            return list(self._ensure().map(fn, items, chunksize=chunksize))
        except Exception as exc:  # BrokenProcessPool and friends
            print(f"      [WARN] pool gagal ({type(exc).__name__}); pool dibuat ulang")
            self.shutdown()
            self.restarts += 1
            try:
                return list(self._ensure().map(fn, items, chunksize=chunksize))
            except Exception as exc2:
                print(f"      [WARN] pool gagal lagi ({type(exc2).__name__}); persona ini serial")
                self.shutdown()
                return [fn(item) for item in items]

    def shutdown(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None


def parse_persona(persona_dir: Path, out_path: Path, pool: WorkerPool,
                  model_name: str, group_folder: str, persona_name: str) -> Counter:
    """Parse one persona folder using the shared, self-healing pool."""
    files = sorted(p for p in persona_dir.glob("*.json") if "warmup" not in p.parts)
    stats: Counter = Counter()
    if not files:
        return stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    worker = partial(parse_one, model_name=model_name, group_folder=group_folder,
                     persona_name=persona_name)
    results = pool.map(worker, files)

    for record in results:
        if record is None:
            stats["unreadable"] += 1
            continue
        if "_error" in record:
            stats["unreadable"] += 1
            print(f"      [WARN] {record['_error']}")
            continue
        records.append(record)
        stats["total"] += 1
        stats[record["answer_status"]] += 1
        stats[f"src::{record['parse_source']}"] += 1
        if record["is_correct"] is True:
            stats["correct"] += 1
        if record["numeric_semantic_conflict"]:
            stats["conflict"] += 1
        if record["candidate_answer"]:
            stats["candidate"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def _print_persona_summary(persona: str, stats: Counter) -> None:
    total = stats["total"] or 1
    committed = stats["committed"]
    print(
        f"      {persona:<28} {stats['total']:>5} soal | "
        f"committed {committed:>5} ({100 * committed / total:5.1f}%) | "
        f"abstain {stats['abstain']:>5} ({100 * stats['abstain'] / total:5.1f}%) | "
        f"missing {stats['missing']:>4}"
    )
    sources = [(k[5:], v) for k, v in stats.items() if k.startswith("src::") and k != "src::none"]
    sources.sort(key=lambda kv: -kv[1])
    if sources:
        top = ", ".join(f"{name} {count}" for name, count in sources[:4])
        print(f"        parse_source: {top}")
    if stats["candidate"]:
        print(f"        [INFO] {stats['candidate']} kandidat semantik (belum dikomit)")
    if stats["conflict"]:
        print(f"        [WARN] {stats['conflict']} baris numeric_semantic_conflict")
    if stats["unreadable"]:
        print(f"        [WARN] {stats['unreadable']} file tidak terbaca")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse ETHICS v9 responses per persona folder")
    ap.add_argument("--group", default="gender",
                    help="gender | religion | regional | neutral (default: gender)")
    ap.add_argument("--all", action="store_true", help="process every persona group")
    ap.add_argument("--model", default=None, help="single model slug")
    ap.add_argument("--persona", default=None, help="single persona folder")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-parse personas already cached")
    args = ap.parse_args()

    slug_to_folder = {slug: folder for folder, slug in GROUPS.items()}
    if args.all:
        folders = list(GROUPS.keys())
    else:
        if args.group not in slug_to_folder:
            ap.error(f"--group must be one of {sorted(slug_to_folder)}")
        folders = [slug_to_folder[args.group]]

    if not RESULT_ROOT.is_dir():
        ap.error(f"result tree not found: {RESULT_ROOT}")

    started = time.time()
    grand: Counter = Counter()

    # One pool for the whole run, rebuilt on demand if a worker dies.
    pool = WorkerPool(args.workers)
    try:
        _run(folders, args, pool, grand)
    finally:
        pool.shutdown()
        if pool.restarts:
            print(f"  [INFO] pool dibuat ulang {pool.restarts}x selama run")

    elapsed = time.time() - started
    total = grand["total"] or 1
    print(f"\n{'=' * 78}")
    print(f"  SELESAI dalam {elapsed / 60:.1f} menit")
    print(f"  total {grand['total']} | committed {grand['committed']} "
          f"({100 * grand['committed'] / total:.1f}%) | abstain {grand['abstain']} "
          f"| missing {grand['missing']}")
    print(f"  kandidat semantik {grand['candidate']} | konflik {grand['conflict']}")
    print(f"  cache: {PARSE_CACHE_ROOT}")
    print(f"{'=' * 78}\n")


def _run(folders, args, pool, grand: Counter) -> None:
    for folder in folders:
        group_dir = RESULT_ROOT / folder
        if not group_dir.is_dir():
            print(f"[SKIP] {folder}: tidak ada")
            continue
        print(f"\n{'=' * 78}\n  {folder}\n{'=' * 78}")

        models = [d for d in sorted(group_dir.iterdir()) if d.is_dir()]
        if args.model:
            models = [d for d in models if d.name == args.model]
            if not models:
                print(f"[SKIP] model {args.model} tidak ada di {folder}")
                continue

        for model_dir in models:
            print(f"\n  [MODEL] {model_dir.name}")
            personas = [(d.name, d) for d in sorted(model_dir.iterdir())
                        if d.is_dir() and d.name != "warmup"]

            # Flat layout: results written straight into the model folder with
            # no persona subdirectory (deepseek's Neutral Persona). Without this
            # the model would be skipped without a word.
            if not personas:
                loose = [p for p in model_dir.glob("*.json")]
                if loose:
                    name = DEFAULT_PERSONA_BY_GROUP.get(folder, model_dir.name)
                    print(f"      [INFO] layout datar terdeteksi ({len(loose)} berkas) "
                          f"-> diperlakukan sebagai persona '{name}'")
                    personas = [(name, model_dir)]
                else:
                    print("      [WARN] tidak ada persona maupun berkas JSON - dilewati")
                    continue

            if args.persona:
                personas = [(n, d) for n, d in personas if n == args.persona]

            for persona_name, persona_dir in personas:
                out_path = PARSE_CACHE_ROOT / folder / model_dir.name / f"{persona_name}.jsonl"
                if out_path.exists() and not args.force:
                    print(f"      {persona_name:<28} [CACHED] (pakai --force untuk ulang)")
                    continue
                stats = parse_persona(persona_dir, out_path, pool,
                                      model_dir.name, folder, persona_name)
                if not stats:
                    print(f"      {persona_name:<28} [KOSONG]")
                    continue
                _print_persona_summary(persona_name, stats)
                grand.update(stats)


if __name__ == "__main__":
    main()
