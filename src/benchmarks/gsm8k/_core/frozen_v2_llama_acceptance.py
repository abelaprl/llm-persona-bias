"""Isolated Llama-3.3 acceptance, blind pilot audit, and technical manifest."""
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .numbers import answers_equal, parse_gold
from ..Religion_Persona.run_frozen_v2 import verify_source_lock

MODEL = "meta-llama_llama-3.3-70b-instruct"
EXPECTED_PERSONAS = 16
EXPECTED_EACH = 1319
EXPECTED_ROWS = EXPECTED_PERSONAS * EXPECTED_EACH
TECHNICAL_FIELDS = [
    "model", "persona", "question_id", "raw_file", "failure_class",
    "response_success", "finish_reason", "provider_error_code",
    "provider_error_message", "provider_error_type", "retry_count_existing",
    "eligible_for_technical_rerun", "reason", "raw_json_sha256",
    "raw_json_valid", "temperature", "max_tokens", "partial_content_length",
]


def atomic_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_one(path, check_splits):
    rows = load(path)
    counts = Counter(r["persona"] for r in rows)
    keys = [(r["persona"], r["question_id"]) for r in rows]
    assert len(rows) == EXPECTED_ROWS
    assert len(counts) == EXPECTED_PERSONAS and set(counts.values()) == {EXPECTED_EACH}
    assert len(keys) == len(set(keys))
    for r in rows:
        expected = answers_equal(r["extracted_answer"] or None,
                                 parse_gold(r["ground_truth_raw"]))
        assert (r["is_correct"] == "True") == expected
        if r["finish_reason"] in ("error", "content_filter"):
            assert r["answer_status"] == "missing_content"
        if r["finish_reason"] == "length":
            if r["answer_status"] == "committed":
                assert r["parse_source"] in ("hashes", "boxed", "marker_line")
    base = path.parent
    if check_splits:
        for persona in counts:
            split = load(base / f"results_{persona}.csv")
            expected = [r for r in rows if r["persona"] == persona]
            assert split == expected
    return rows


def _summary(rows):
    groups = defaultdict(list)
    for r in rows: groups[r["persona"]].append(r)
    groups["OVERALL"] = rows
    out = {}
    for p, rs in groups.items():
        n = len(rs); correct = sum(r["is_correct"] == "True" for r in rs)
        committed = sum(r["answer_status"] == "committed" for r in rs)
        out[p] = (n, correct, committed,
                  round(100 * correct / n, 4), round(100 * committed / n, 4))
    return out


def check_summary(rows, accuracy_path, coverage_path):
    calc = _summary(rows)
    for r in load(accuracy_path):
        n, correct, committed, acc, commit = calc[r["persona"]]
        assert (int(r["total"]), int(r["correct"]), int(r["committed"])) == (n, correct, committed)
        assert float(r["accuracy"]) == acc and float(r["commit_rate"]) == commit
    for r in load(coverage_path):
        n, _, committed, _, commit = calc[r["persona"]]
        assert (int(r["total"]), int(r["committed"])) == (n, committed)
        assert float(r["commit_rate"]) == commit


def blind_audit(rows, out, n_per_stratum=3):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["model"], r["parse_source"] or "none",
                r["priority_hit"] or "none", r["extraction_confidence"] or "none",
                r["finish_reason"] or "none", r["persona"],
                r["answer_status"])].append(r)
    chosen = []
    for key in sorted(strata):
        rng = random.Random("frozen-v2|" + "|".join(key))
        pool = strata[key][:]
        rng.shuffle(pool)
        chosen.extend(pool[:n_per_stratum])
    fields = ["model", "persona", "question_id", "file", "response_content_full",
              "extracted_answer", "candidate_answer", "answer_status", "parse_source",
              "priority_hit", "extraction_confidence", "abstain_reason", "finish_reason",
              "audit_extracted_span", "annotator_valid_without_gt", "audit_notes"]
    atomic_csv(out, fields, chosen)
    return len(chosen)


def check_semantic_replay(path):
    replay = load(path)
    expected = {"results.csv", "results_core.csv"}
    assert {r["file"] for r in replay} == expected
    for r in replay:
        assert int(r["rows"]) == EXPECTED_ROWS
        assert int(r["mismatches"]) == 0
        assert str(r["match"]).lower() == "true"


def main():
    lock = verify_source_lock()
    base = Path(r"result\GMS8K\reports_parser_v2\Religion_Persona") / MODEL
    audit = Path(r"result\GMS8K\audit\frozen_llama33")
    model_path, core_path = base / "results.csv", base / "results_core.csv"
    replay_path = Path(r"result\GMS8K\audit\frozen_v2_semantic_replay.csv")
    check_semantic_replay(replay_path)
    model_before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = validate_one(model_path, True)
    core = validate_one(core_path, False)
    check_summary(model, base / "accuracy.csv", base / "coverage.csv")
    check_summary(core, base / "accuracy_core.csv", base / "coverage_core.csv")
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_before
    tech = [r for r in model if r["finish_reason"] in ("error", "content_filter", "length")]
    errors = [r for r in tech if r["finish_reason"] == "error"]
    lengths = [r for r in tech if r["finish_reason"] == "length"]
    assert len(errors) == 202 and len(lengths) == 23
    raw_root = Path(r"result\GMS8K\Religion_Persona") / MODEL
    technical_units = []
    for r in tech:
        raw_path = raw_root / Path(r["file"])
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
        response = raw.get("response") or {}
        choices = ((response.get("raw_api_response") or {}).get("choices") or [])
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        raw_finish = choice.get("finish_reason", "")
        assert raw_finish == r["finish_reason"]
        provider_error = choice.get("error") or response.get("error") or {}
        if isinstance(provider_error, dict):
            provider_code = provider_error.get("code", "")
            provider_message = provider_error.get("message", "")
            provider_type = (provider_error.get("metadata") or {}).get("error_type", "")
        else:
            provider_code, provider_message, provider_type = "", str(provider_error), ""
        request_parameters = (raw.get("request") or {}).get("parameters") or {}
        performance = response.get("performance") or {}
        content = response.get("content")
        content = content if isinstance(content, str) else ""
        eligible = raw_finish == "error"
        if eligible:
            assert str(provider_code) == "500"
            assert provider_message == "Internal Server Error"
        unit = {
            "model": MODEL,
            "persona": r.get("persona", ""),
            "question_id": r.get("question_id", ""),
            "raw_file": str(raw_path),
            "failure_class": ("finish_reason_error" if eligible else
                              "finish_reason_length"),
            "response_success": response.get("success") is True,
            "finish_reason": raw_finish,
            "provider_error_code": provider_code,
            "provider_error_message": provider_message,
            "provider_error_type": provider_type,
            "retry_count_existing": performance.get("retry_count", ""),
            "eligible_for_technical_rerun": eligible,
            "reason": ("provider 500 technical failure" if eligible else
                       "length is an experimental outcome; excluded from technical rerun"),
            "raw_json_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_json_valid": True,
            "temperature": request_parameters.get("temperature", ""),
            "max_tokens": request_parameters.get("max_tokens", ""),
            "partial_content_length": len(content),
            "file": r.get("file", ""),
            "answer_status": r.get("answer_status", ""),
            "abstain_reason": r.get("abstain_reason", ""),
        }
        technical_units.append(unit)
    manifest = {
        "model": MODEL, "no_api": True, "source_lock": lock,
        "counts": {"rows_model": len(model), "rows_core": len(core),
                   "finish_error": len(errors), "finish_length_ineligible": len(lengths)},
        "technical_units": technical_units,
    }
    audit.mkdir(parents=True, exist_ok=True)
    tmp = audit / ".technical_rerun_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, audit / "technical_rerun_manifest.json")
    atomic_csv(Path(r"result\GMS8K\audit\religion_persona_technical_rerun_manifest.csv"),
               TECHNICAL_FIELDS, technical_units)
    n_audit = blind_audit(model, audit / "extraction_audit_blind_pilot.csv")
    rows = [{"gate": "model_rows", "value": len(model), "pass": True},
            {"gate": "core_rows", "value": len(core), "pass": True},
            {"gate": "semantic_replay_mismatch", "value": 0, "pass": True},
            {"gate": "finish_error", "value": len(errors), "pass": True},
            {"gate": "length_ineligible", "value": len(lengths), "pass": True},
            {"gate": "blind_audit_rows", "value": n_audit, "pass": True}]
    atomic_csv(audit / "acceptance_summary.csv", ["gate", "value", "pass"], rows)
    print(json.dumps({"accepted": True, "model_rows": len(model), "core_rows": len(core),
                      "errors": len(errors), "length_ineligible": len(lengths),
                      "blind_audit_rows": n_audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
