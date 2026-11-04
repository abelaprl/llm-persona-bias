"""Common gender/religion measurements, retaining the original analyses.

Run from the experiment repository:
python -m src.benchmarks.ethics.harmonize_identity_markers --phase all
No generation, answer scoring, or overwriting of previous result directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

ROOT = Path(os.environ.get("TA_EXPERIMENT_ROOT", "D:/TugasAkhir"))
sys.path.insert(0, str(ROOT))
from src.benchmarks.ethics.gender_lm_v2.surfaces import build_surfaces
from src.benchmarks.ethics.gender_lm_v2.detectors import detect_markers
from src.benchmarks.ethics.religion_lm_v3_extract import extract_final
from src.benchmarks.ethics.religion_lm_v3_surface import prepare_final_surface
from src.core.linguistic_markers.registry import MarkerSpec, PatternSpec

VERSION = "identity-common-20261015-v2"
RUNS = ROOT / "result/analyses/linguistic_markers/runs"
SOURCES = {
    "gender_gsm8k": "gender_gsm8k_raw_rebuild_20261004_220122_288416",
    "gender_ethics": "gender_ethics_raw_rebuild_20261004_213224_141860",
    "religion_gsm8k": "religion_gsm8k_raw_rebuild_20261006_001628_358229",
    "religion_ethics": "religion_ethics_raw_rebuild_20261006_003454_929593",
}
GENDER_IDS = ["gender_persona_direct_address", "gender_persona_acknowledgment",
              "gender_second_person_identity_reference", "gender_first_person_identity_reference",
              "gender_perspective_adoption"]
RELIGION_IDS = ["religion_persona_direct_address", "religion_perspective_adoption",
                "religion_moral_authority_framing", "religion_secular_humanist_framing",
                "religion_pluralist_framing", "religion_identity_irrelevance",
                "religion_blessing_invocation"]
PRONOUNS = {"masculine_pronouns": re.compile(r"\b(?:he|him|his|himself)\b", re.I),
            "feminine_pronouns": re.compile(r"\b(?:she|her|hers|herself)\b", re.I)}
# Observed in full GSM8K readback rows 17237, 18197, 18317, 39978, 40359.
# This is an identity-qualified denial, not adoption of a religious perspective.
IRRELEVANCE_GUARD = re.compile(
    r"\bas\s+(?:a|an)\s+(?:atheist|christian|muslim|jew|jewish|agnostic)\s*,?\s+"
    r"your\s+(?:worldview|beliefs?)\s+(?:doesn't|don't|does\s+not|do\s+not)\s+affect\b", re.I)
LABELS = {
    "masculine_pronouns": "Pronomina laki-laki", "feminine_pronouns": "Pronomina perempuan",
    "gender_persona_direct_address": "Sapaan gender pengguna",
    "gender_persona_acknowledgment": "Pengakuan identitas gender pengguna",
    "gender_second_person_identity_reference": "Rujukan gender pengguna",
    "gender_first_person_identity_reference": "Pengulangan identitas orang pertama",
    "gender_perspective_adoption": "Sudut pandang gender",
    "religion_persona_direct_address": "Sapaan agama pengguna",
    "religion_perspective_adoption": "Sudut pandang agama",
    "religion_moral_authority_framing": "Alasan dari ajaran agama",
    "religion_secular_humanist_framing": "Alasan sekuler atau humanis",
    "religion_pluralist_framing": "Perbandingan pandangan antaragama",
    "religion_identity_irrelevance": "Agama tidak berkaitan dengan soal",
    "religion_blessing_invocation": "Doa atau berkat",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def registry(domain):
    items = json.loads((RUNS / SOURCES[domain + "_ethics"] / "registry.json").read_text(encoding="utf-8"))
    ids = GENDER_IDS if domain == "gender" else RELIGION_IDS
    return [next(m for m in items if m["marker_id"] == marker) for marker in ids]


_REG = {}


def specs(domain):
    if domain not in _REG:
        r = registry(domain)
        if domain == "religion":
            r = [MarkerSpec(marker_id=m["marker_id"], family=m["family"],
                            operational_definition=m["operational_definition"],
                            patterns=tuple(PatternSpec(x["pattern"], x.get("note", ""), x.get("flags", re.I)) for x in m["patterns"]),
                            measurement=m["measurement"], surface=m["surface"],
                            benchmarks=("gsm8k", "ethics"),
                            exclusion_rule=m.get("exclusion_rule", "")) for m in r]
        _REG[domain] = r
    return _REG[domain]


def measure_one(task):
    domain, raw = task
    # Existing final-only adapter masks think blocks and selects explicit Harmony final.
    base = build_surfaces(raw)
    visible = base.visible_text
    available = bool(visible.strip()) and base.status != "unknown_surface"
    output = {"visible_available": available, "surface_status": base.status,
              "visible_sha256": hashlib.sha256(visible.encode()).hexdigest()}
    spans = []
    if domain == "gender":
        output.update(explanation_available=base.status == "substantive_explanation", word_count=base.word_count)
        for mid, pattern in PRONOUNS.items():
            hits = list(pattern.finditer(visible)) if available else []
            output[mid + "__count"] = len(hits) if available else np.nan
            output[mid + "__present"] = int(bool(hits)) if available else np.nan
            spans += [{"marker_id": mid, "rule_id": "closed_pronoun_lexicon", "text": h.group(),
                       "start": h.start(), "end": h.end(), "surface": "visible"} for h in hits]
        hits = detect_markers(base, specs(domain))
        counts = Counter(h.marker_id for h in hits)
        for mid in GENDER_IDS:
            n = counts[mid] if output["explanation_available"] else np.nan
            output[mid + "__count"] = n
            output[mid + "__present"] = int(n > 0) if np.isfinite(n) else np.nan
        spans += [{"marker_id": h.marker_id, "rule_id": h.rule_id, "text": h.text,
                   "start": h.start, "end": h.end, "surface": h.surface_name} for h in hits]
    else:
        measured, hits = extract_final(visible, registry=specs(domain))
        if re.search(r"\b(?:worldview|beliefs?)\b", visible, re.I):
            surface = prepare_final_surface(visible)
            corrections = list(IRRELEVANCE_GUARD.finditer(surface.linguistic_text))
            for match in corrections:
                for hit in hits:
                    if hit['marker_id'] == 'religion_perspective_adoption' and match.start() <= hit['start'] < match.end():
                        hit.update(candidate_status='excluded', exclusion_reason='identity_qualified_denial_of_relevance')
                if not any(h['marker_id'] == 'religion_identity_irrelevance' and h['candidate_status'] != 'excluded'
                           and h['start'] < match.end() and h['end'] > match.start() for h in hits):
                    hits.append(dict(marker_id='religion_identity_irrelevance', rule_id='identity_qualified_denial_of_relevance',
                                     matched_text=match.group(), start=match.start(), end=match.end(),
                                     surface_name='lexical', candidate_status='accepted'))
            if corrections:
                for mid in ['religion_perspective_adoption', 'religion_identity_irrelevance']:
                    count = sum(h['marker_id'] == mid and h['candidate_status'] != 'excluded' for h in hits)
                    measured[mid+'__count'] = count
                    measured[mid+'__present'] = int(count > 0)
                    measured[mid+'__density'] = count / measured['word_count'] * 100
        output.update(explanation_available=measured["explanation_available"], word_count=measured["word_count"])
        for mid in RELIGION_IDS:
            for suffix in ["count", "present", "density"]:
                value = measured[mid + "__" + suffix]
                output[mid + "__" + suffix] = np.nan if value is None else value
        spans = [{"marker_id": h["marker_id"], "rule_id": h["rule_id"], "text": h["matched_text"],
                  "start": h["start"], "end": h["end"], "surface": h["surface_name"]}
                 for h in hits if h["candidate_status"] != "excluded"]
    return output, spans


def outcomes(domain):
    r = registry(domain)
    ids = list(PRONOUNS) + GENDER_IDS if domain == "gender" else RELIGION_IDS
    result = [(m, "present") for m in ids]
    result += [(m["marker_id"], "density") for m in r if m["measurement"] == "density"]
    return result


def measure_all(out, workers):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "sources": {}, "seed": 20261015, "bootstrap_resamples": 2000,
                "family_policy": "all common outcomes x all contrasts x all scopes, per model/domain/task",
                "prior_results_modified": False, "code_sha256": sha(__file__)}
    (out / "instrument_gender.json").write_text(json.dumps(registry("gender"), indent=2), encoding="utf-8")
    (out / "instrument_religion.json").write_text(json.dumps(registry("religion"), indent=2), encoding="utf-8")
    (out / "pronouns.json").write_text(json.dumps({m: p.pattern for m,p in PRONOUNS.items()}, indent=2), encoding="utf-8")
    (out / "context_guard.json").write_text(json.dumps({'version':VERSION,'pattern':IRRELEVANCE_GUARD.pattern,
        'source':'Full GSM8K readback, rows 17237, 18197, 18317, 39978, 40359',
        'scope':'Both benchmarks, all models/personas; no score/label inputs'}, indent=2), encoding='utf-8')
    audit = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, source in SOURCES.items():
            domain, benchmark = name.split("_")
            path = RUNS / source / "response_records.csv.gz"
            manifest["sources"][name] = {"path": str(path), "sha256": sha(path)}
            frames = []
            keep = ["model_id", "persona", "subset", "question_id", "question_hash", "question", "raw_source_file",
                    "raw_sha256", "pair_eligible", "raw_final_text", "final_text", "reasoning_used"]
            row_number = 0
            for chunk in pd.read_csv(path, usecols=lambda c: c in keep, keep_default_na=False, chunksize=4000):
                assert not chunk.reasoning_used.astype(str).str.lower().eq("true").any()
                text_col = "raw_final_text" if "raw_final_text" in chunk else "final_text"
                records = chunk.to_dict("records")
                values = list(pool.map(measure_one, ((domain, r[text_col]) for r in records), chunksize=80))
                for original, (measured, spans) in zip(records, values):
                    row_number += 1
                    raw = original.pop(text_col)
                    original.pop("reasoning_used")
                    original["pair_eligible"] = str(original["pair_eligible"]).lower() == "true"
                    original["source_row"] = row_number
                    original["snapshot_final_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
                    original.update(measured)
                    for mid in {h["marker_id"] for h in spans}:
                        key = (name, original["model_id"], mid)
                        rank = hashlib.sha256((name + str(row_number) + mid).encode()).hexdigest()
                        bucket = audit[key]
                        if len(bucket) < 10 or rank < max(x["sample_rank"] for x in bucket):
                            bucket.append(dict(original, dataset=name, marker_id=mid, sample_rank=rank,
                                               full_response=raw, hits=[h for h in spans if h["marker_id"] == mid]))
                            bucket.sort(key=lambda x: x["sample_rank"])
                            del bucket[10:]
                    original.pop("question")
                frames.append(pd.DataFrame(records))
                print(name, row_number, flush=True)
            frame = pd.concat(frames, ignore_index=True)
            assert not frame.loc[frame.pair_eligible].duplicated(["model_id", "persona", "subset", "question_id", "question_hash"]).any()
            frame.to_csv(out / (name + "_records.csv.gz"), index=False)
            manifest["sources"][name]["rows"] = len(frame)
            print("Saved", name, len(frame), flush=True)
    with (out / "audit_samples.jsonl").open("w", encoding="utf-8") as f:
        for key in sorted(audit):
            for sample in audit[key]:
                f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    manifest["audit_sample_rows"] = sum(map(len, audit.values()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def contrasts(personas):
    identities = sorted({re.sub(r"_(?:explicit|implicit)_?[123]?$", "", p) for p in personas if p != "neutral"})
    mapping = {}
    for identity in identities:
        for condition in ["E", "I1", "I2", "I3"]:
            candidates = [p for p in personas if p.startswith(identity + "_") and
                          ((condition == "E" and "explicit" in p) or (condition != "E" and p.endswith("implicit_" + condition[1:])))]
            assert len(candidates) == 1, (identity, condition, candidates)
            mapping[identity, condition] = candidates[0]
    pairs = []
    for identity in identities:
        for c in ["E", "I1", "I2", "I3"]:
            pairs.append((mapping[identity,c], "neutral", "persona_vs_N"))
        for c in ["I1", "I2", "I3"]:
            pairs.append((mapping[identity,c], mapping[identity,"E"], "I_vs_E"))
    for i, left in enumerate(identities):
        for right in identities[i+1:]:
            for c in ["E", "I1", "I2", "I3"]:
                pairs.append((mapping[left,c], mapping[right,c], "between_personas"))
    assert len(pairs) == (18 if len(identities) == 2 else 52)
    return pairs


def holm(pvalues):
    p = np.array([1.0 if not np.isfinite(x) else x for x in pvalues])
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    adjusted[~np.isfinite(pvalues)] = np.nan
    return adjusted


def estimate(a, b, binary, rng):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n == 0:
        return dict(n_pairs=0, p_raw=np.nan, delta=np.nan, ci_low=np.nan, ci_high=np.nan,
                    effect_size=np.nan, effect_name="Cohen h" if binary else "matched rank-biserial", status="unavailable")
    delta = a - b
    nonzero = delta[delta != 0]
    result = dict(n_pairs=n, n_nonzero=len(nonzero), mean_left=a.mean(), mean_right=b.mean(),
                  delta=delta.mean() * (100 if binary else 1), status="no_change" if not len(nonzero) else "estimated")
    if binary:
        plus = int(((a == 1) & (b == 0)).sum()); minus = int(((a == 0) & (b == 1)).sum())
        p = binomtest(plus, plus+minus, .5).pvalue if plus+minus else 1.0
        effect = 2 * (np.arcsin(np.sqrt(a.mean())) - np.arcsin(np.sqrt(b.mean())))
        result.update(discordant_left_only=plus, discordant_right_only=minus, effect_name="Cohen h")
    else:
        p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="approx").pvalue if len(nonzero) else 1.0
        ranks = rankdata(abs(nonzero))
        effect = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum()) if len(nonzero) else np.nan
        result["effect_name"] = "matched rank-biserial"
    # Multinomial bootstrap of the empirical paired-difference distribution is
    # exactly equivalent to resampling paired rows for a mean-difference CI.
    unique, counts = np.unique(delta, return_counts=True)
    if len(unique) == 1:
        lo = hi = unique[0]
    else:
        draws = rng.multinomial(n, counts/n, size=2000)
        means = np.sum(draws * unique[None, :], axis=1) / n
        lo, hi = np.quantile(means, [.025,.975])
    result.update(p_raw=float(p), effect_size=effect, ci_low=lo*(100 if binary else 1),
                  ci_high=hi*(100 if binary else 1), ci_method="paired_percentile_bootstrap_2000")
    return result


def statistics(out):
    stats, profiles = [], []
    rng = np.random.default_rng(20261015)
    for name in SOURCES:
        domain, benchmark = name.split("_")
        frame = pd.read_csv(out / (name + "_records.csv.gz"), keep_default_na=True)
        scopes = sorted(frame["subset"].unique())
        if benchmark == "ethics":
            scopes += ["ethics_all"]
        pairs = contrasts(frame.persona.unique())
        for model in sorted(frame.model_id.unique()):
            for scope in scopes:
                group = frame[frame.model_id.eq(model)]
                if scope != "ethics_all":
                    group = group[group["subset"].eq(scope)]
                for mid, unit in outcomes(domain):
                    col = mid + "__" + unit
                    for persona, part in group.groupby("persona"):
                        values = part[col].dropna()
                        count = part[mid + "__count"].dropna()
                        profiles.append(dict(dataset=name, model_id=model, subset=scope, persona=persona,
                            marker_id=mid, measurement=unit, total=len(part), n_available=len(values),
                            mean=float(values.mean()) if len(values) else np.nan,
                            raw_occurrences=float(count.sum()) if len(count) else np.nan,
                            n_responses_present=int(count.gt(0).sum()) if len(count) else 0))
                eligible = group[group.pair_eligible.eq(True)]
                for left, right, comparison in pairs:
                    cols = ["subset", "question_id", "question_hash"] + [m + "__" + u for m,u in outcomes(domain)]
                    paired = eligible[eligible.persona.eq(left)][cols].merge(
                        eligible[eligible.persona.eq(right)][cols], on=["subset", "question_id", "question_hash"], suffixes=("_left", "_right"), validate="one_to_one")
                    for mid, unit in outcomes(domain):
                        result = estimate(paired[mid+"__"+unit+"_left"].to_numpy(float),
                                          paired[mid+"__"+unit+"_right"].to_numpy(float), unit == "present", rng)
                        stats.append(dict(dataset=name, model_id=model, subset=scope, left=left, right=right,
                                          comparison=comparison, marker_id=mid, measurement=unit, **result))
            print("Statistics", name, model, flush=True)
    stats = pd.DataFrame(stats)
    for (name, model), indices in stats.groupby(["dataset", "model_id"]).groups.items():
        expected = len(outcomes(name.split("_")[0])) * (18 if name.startswith("gender") else 52) * (5 if name.endswith("ethics") else 1)
        assert len(indices) == expected, (name, len(indices), expected)
        stats.loc[indices,"p_holm"] = holm(stats.loc[indices,"p_raw"].to_numpy(float))
        stats.loc[indices,"holm_family_size"] = expected
    stats.to_csv(out / "paired_statistics.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "profiles.csv", index=False)
    summary = stats.groupby(["dataset", "marker_id", "measurement"]).agg(
        tests=("p_raw", "size"), available=("p_raw", "count"), significant=("p_holm", lambda x: int(x.lt(.05).sum())),
        min_delta=("delta", "min"), max_delta=("delta", "max"))
    summary.to_csv(out / "summary.csv")
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["measure", "stats", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output or RUNS / ("identity_common_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.phase in ["all", "measure"]:
        measure_all(out, args.workers)
    if args.phase in ["all", "stats"]:
        statistics(out)
    print("RESULT", out, flush=True)


if __name__ == "__main__":
    main()
