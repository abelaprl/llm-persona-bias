"""Adapter display-only untuk artefak beku Regional GSM8K dan ETHICS.

Modul ini tidak mengekstrak marker, menilai jawaban, atau menghitung inferensi
baru. Agregasi yang dilakukan hanya menata ulang count baris beku agar dua
benchmark dapat memakai kontrak visual yang sama.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
GSM_RUN = (ROOT / "result/benchmarks/gsm8k/reports/Regional_Cultural_Persona/linguistic_markers"
           / "regional_gsm8k_full_source_aligned_v1_1_20260818")
GSM_VISUAL = (ROOT / "result/benchmarks/gsm8k/reports/Regional_Cultural_Persona/linguistic_markers/visual"
              / "regional_gsm8k_bab4_visual_v1_1_0_20260818")
ETHICS_BUNDLE = ROOT / "result/benchmarks/ethics/reports/regional_ethics_presentation_20260919_r2"
ETHICS_CANONICAL = (ROOT / "result/benchmarks/ethics/reports/Regional_Cultural_Persona/_combined"
                    / "all_models_results_full.csv")

MODELS = [
    "deepseek_deepseek-r1-distill-qwen-32b",
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
]
IDENTITIES = ["north_american", "east_asian", "south_asian", "sub_saharan_african"]
CONDITIONS = ["explicit", "implicit_1", "implicit_2", "implicit_3"]
SCENARIOS = {"neutral": "N", "explicit": "E", "implicit_1": "I1",
             "implicit_2": "I2", "implicit_3": "I3"}
IDENTITY_LABELS = {
    "north_american": "Amerika Utara",
    "east_asian": "Asia Timur",
    "south_asian": "Asia Selatan",
    "sub_saharan_african": "Afrika Sub-Sahara",
}
PERSONAS = ["neutral"] + [f"{identity}_{condition}"
                            for identity in IDENTITIES for condition in CONDITIONS]
PERSONA_LABELS = {"neutral": "N"}
for _identity in IDENTITIES:
    for _condition in CONDITIONS:
        PERSONA_LABELS[f"{_identity}_{_condition}"] = (
            f"{IDENTITY_LABELS[_identity]} {SCENARIOS[_condition]}"
        )

# Seluruhnya density di kedua registry. Dua marker regional lain sengaja tidak
# masuk core karena unitnya density di GSM8K tetapi binary di ETHICS.
CORE_MARKERS = [
    "culture_region_north_american",
    "culture_region_east_asian",
    "culture_region_south_asian",
    "culture_region_sub_saharan_african",
    "culture_general_reference",
    "culture_meta_framing",
]
CORE_LABELS = {
    "culture_region_north_american": "Rujukan Amerika Utara",
    "culture_region_east_asian": "Rujukan Asia Timur",
    "culture_region_south_asian": "Rujukan Asia Selatan",
    "culture_region_sub_saharan_african": "Rujukan Afrika Sub-Sahara",
    "culture_general_reference": "Rujukan budaya umum",
    "culture_meta_framing": "Meta-framing budaya",
}
CATALOG_LABELS = {
    'culture_geocultural_broader_reference': 'Rujukan geobudaya',
    'culture_relativist_framing': 'Relativisme budaya',
    'regional_persona_direct_address': 'Sapaan persona',
    'regional_perspective_adoption': 'Adopsi perspektif',
    'culture_collectivist_communal_orientation': 'Kolektivisme',
    'culture_individual_autonomy_orientation': 'Otonomi individu',
    'culture_family_obligation_orientation': 'Kewajiban keluarga',
    'culture_community_obligation_orientation': 'Kewajiban komunitas',
    'culture_reciprocity_interdependence_orientation': 'Resiprositas',
    'culture_hospitality_generosity_orientation': 'Keramahan / kemurahan',
    'culture_elder_respect_orientation': 'Hormat pada sesepuh',
    'culture_harmony_face_orientation': 'Harmoni / menjaga muka',
    'culture_hierarchy_authority_orientation': 'Hierarki / otoritas',
    'culture_named_concept_reference': 'Konsep budaya bernama',
    'culture_named_practice_reference': 'Praktik budaya bernama',
    'culture_warmup_name_echo': 'Gema nama warmup',
    'culture_identity_irrelevance': 'Identitas tak relevan',
}
ETHICS_POLICY = "regional-explanation-v1-pairing-v2-20260919"

FILES = {
    "GSM8K": {
        "manifest": GSM_RUN / "manifest.json",
        "catalog": GSM_RUN / "marker_registry.csv",
        "length": GSM_RUN / "condition_descriptives.csv",
        "effects": GSM_RUN / "paired_numeric.csv",
        "records": GSM_RUN / "response_records.csv",
        "coverage": GSM_VISUAL / "response_counts_model_persona.csv",
        "terms": GSM_VISUAL / "regional_term_prevalence_model_persona.csv",
    },
    "ETHICS": {
        "verification": ETHICS_BUNDLE / "verification.json",
        "catalog": ETHICS_BUNDLE / "tables/marker_catalog.csv",
        "coverage": ETHICS_BUNDLE / "tables/coverage.csv",
        "profiles": ETHICS_BUNDLE / "tables/all_marker_profiles.csv",
        "effects": ETHICS_BUNDLE / "tables/statistics/paired_contrasts.csv.gz",
        "statistics_provenance": ETHICS_BUNDLE / "tables/statistics/provenance.json",
        "pairing_exclusions": ETHICS_BUNDLE / "tables/pairing_exclusions.csv",
        "analysis_view": ETHICS_BUNDLE / "tables/analysis_view.csv.gz",
        "terms": ETHICS_BUNDLE / "evidence/context_hits.csv.gz",
        "records": ETHICS_CANONICAL,
    },
}


def input_paths() -> list[Path]:
    """Semua input yang benar-benar dibaca adapter, termasuk denominator/teks."""
    return [path for benchmark in ("GSM8K", "ETHICS")
            for path in FILES[benchmark].values()]


def _identity_condition(persona: str) -> tuple[str, str]:
    if persona == "neutral":
        return "neutral", "neutral"
    for condition in CONDITIONS:
        suffix = "_" + condition
        if persona.endswith(suffix):
            identity = persona[:-len(suffix)]
            if identity in IDENTITIES:
                return identity, condition
    raise ValueError(f"Persona Regional tidak dikenal: {persona}")


def _assert_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        example = frame.loc[frame.duplicated(keys, keep=False), keys].iloc[0].to_dict()
        raise ValueError(f"Duplikat {name}: {example}")


def adapt_effects(source: pd.DataFrame, benchmark: str, detail: bool = False) -> pd.DataFrame:
    """Tambahkan kolom display tanpa mengubah kolom inferensi sumber.

    Overview selalu paired rank-biserial tanpa CI. Detail selalu median
    paired difference beserta CI bootstrap median yang memang se-estimand.
    """
    if benchmark not in {"GSM8K", "ETHICS"}:
        raise ValueError(f"Benchmark tidak dikenal: {benchmark}")
    out = source.copy(deep=True)
    required = {"model_id", "identity", "contrast_id", "left_condition",
                "right_condition", "left_persona", "right_persona", "outcome_id",
                "n_pairs", "n_nonzero", "ci_low", "ci_high", "p_holm"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Kolom effect hilang: {sorted(missing)}")
    expected_contrast = out.left_condition.astype(str) + "_vs_neutral"
    expected_left = out.identity.astype(str) + "_" + out.left_condition.astype(str)
    if not (out.right_condition.eq("neutral") & out.right_persona.eq("neutral")
            & out.contrast_id.eq(expected_contrast) & out.left_persona.eq(expected_left)).all():
        raise ValueError("Arah kontras harus persona minus neutral")

    if benchmark == "GSM8K":
        if "rank_biserial" not in out or "median_paired_delta" not in out:
            raise ValueError("Estimand GSM8K tidak lengkap")
        rank = pd.to_numeric(out.rank_biserial, errors="coerce")
        median = pd.to_numeric(out.median_paired_delta, errors="coerce")
    else:
        checks = (
            out.get("population", pd.Series(index=out.index, dtype=object)).eq("text_all")
            & out.get("registered_measurement", pd.Series(index=out.index, dtype=object)).eq("density")
            & out.get("outcome_scale", pd.Series(index=out.index, dtype=object)).eq("density_per_100_words")
            & out.get("effect_size_name", pd.Series(index=out.index, dtype=object)).eq("paired_rank_biserial")
            & out.get("analysis_policy_version", pd.Series(index=out.index, dtype=object)).eq(ETHICS_POLICY)
        )
        expected_direction = out.left_condition.astype(str) + "_minus_neutral"
        if not checks.all() or not out.direction.eq(expected_direction).all():
            raise ValueError("arah, unit, populasi, atau policy ETHICS tidak sah")
        rank = pd.to_numeric(out.effect_size, errors="coerce")
        median = pd.to_numeric(out.effect, errors="coerce")

    out["marker_id"] = out.outcome_id
    out["row_id"] = out.outcome_id
    out["scenario"] = out.left_condition.map(SCENARIOS)
    if out.scenario.isna().any():
        raise ValueError("Skenario effect tidak dikenal")
    out["panel"] = out.identity
    out["changed"] = pd.to_numeric(out.n_nonzero, errors="coerce")
    out["corrected_p"] = pd.to_numeric(out.p_holm, errors="coerce")
    if detail:
        out["estimate"] = median
        out["lower"] = pd.to_numeric(out.ci_low, errors="coerce")
        out["upper"] = pd.to_numeric(out.ci_high, errors="coerce")
        out["estimand_display"] = "median_paired_difference"
        out["display_unit"] = "occurrences_per_100_words"
        supported = out.n_pairs.gt(0) & out.estimate.notna()
        if (out.loc[supported, "lower"].isna().any()
                or out.loc[supported, "upper"].isna().any()
                or (out.loc[supported, "lower"] > out.loc[supported, "estimate"]).any()
                or (out.loc[supported, "estimate"] > out.loc[supported, "upper"]).any()):
            raise ValueError("CI detail bukan CI median yang valid")
    else:
        out["estimate"] = rank
        out["lower"] = np.nan
        out["upper"] = np.nan
        out["estimand_display"] = "paired_rank_biserial"
        out["display_unit"] = "rank_biserial"
    _assert_unique(out, ["model_id", "identity", "scenario", "row_id"], "effect")
    return out


def build_terms(source: pd.DataFrame, denominators: pd.DataFrame,
                models: list[str], identities: list[str], top_n: int = 6) -> pd.DataFrame:
    """Pilih top-N phrase per identity; tie diurutkan alfabetis."""
    needed = {"model_id", "persona", "term", "raw_count"}
    if needed - set(source) or {"model_id", "persona", "total_words", "n_available"} - set(denominators):
        raise ValueError("Skema phrase/denominator tidak lengkap")
    counts = (source.groupby(["model_id", "persona", "term"], as_index=False,
                             dropna=False).raw_count.sum())
    denom = denominators.copy()
    _assert_unique(denom, ["model_id", "persona"], "denominator phrase")
    records: list[dict[str, object]] = []
    for identity in identities:
        persona = identity + "_explicit"
        ranked = (counts[counts.persona.eq(persona)].groupby("term", as_index=False)
                  .raw_count.sum().loc[lambda frame: frame.raw_count.gt(0)]
                  .sort_values(["raw_count", "term"], ascending=[False, True],
                               kind="mergesort").head(top_n))
        terms = ranked.term.tolist()
        terms += [f"(tidak tersedia {order})"
                  for order in range(len(terms) + 1, top_n + 1)]
        for model, (order, term) in product(models, enumerate(terms, 1)):
            records.append(dict(model_id=model, persona=persona, identity=identity,
                                scenario="E", term=term, row_id=term, term_order=order,
                                term_available=not term.startswith("(tidak tersedia ")))
    out = pd.DataFrame(records)
    out = out.merge(counts, on=["model_id", "persona", "term"], how="left",
                    validate="one_to_one")
    out = out.merge(denom, on=["model_id", "persona"], how="left",
                    validate="many_to_one")
    valid = out.total_words.gt(0) & out.n_available.gt(0)
    fillable = valid & out.term_available
    out.loc[fillable, "raw_count"] = out.loc[fillable, "raw_count"].fillna(0)
    out.loc[~fillable, "raw_count"] = np.nan
    out["estimate"] = np.where(fillable, 100 * out.raw_count / out.total_words, np.nan)
    out["display_unit"] = "occurrences_per_100_words"
    _assert_unique(out, ["model_id", "identity", "scenario", "row_id"], "phrase")
    return out.sort_values(["identity", "term_order", "model_id"],
                           kind="mergesort").reset_index(drop=True)


def build_examples(source: pd.DataFrame, benchmark: str,
                   identity: str = "north_american") -> pd.DataFrame:
    """Pilih triplet pertama N/E/I1 yang benar-benar satu model dan soal."""
    personas = ["neutral", f"{identity}_explicit", f"{identity}_implicit_1"]
    panels = dict(zip(personas, ["N", "E", "I1"]))
    needed = {"model_id", "persona", "question_id", "response_id", "full_text", "source_path"}
    if needed - set(source):
        raise ValueError("Skema contoh respons tidak lengkap")
    rows = source[source.persona.isin(personas)].copy()
    if "pairing_eligible" in rows:
        rows = rows[rows.pairing_eligible.fillna(False).astype(bool)]
    rows = rows[rows.full_text.fillna("").astype(str).str.strip().ne("")]
    _assert_unique(rows, ["model_id", "persona", "question_id"], "contoh respons")
    support = (rows.groupby(["model_id", "question_id"]).persona
               .agg(lambda values: set(values)).reset_index())
    support = support[support.persona.map(lambda values: set(personas).issubset(values))]
    if support.empty:
        raise ValueError("Tidak ada triplet N/E/I1 dengan model dan soal yang sama")
    model_rank = {model: i for i, model in enumerate(MODELS)}
    support["model_rank"] = support.model_id.map(model_rank).fillna(len(model_rank))
    chosen = support.sort_values(["model_rank", "question_id", "model_id"],
                                 kind="mergesort").iloc[0]
    out = rows[rows.model_id.eq(chosen.model_id) & rows.question_id.eq(chosen.question_id)
               & rows.persona.isin(personas)].copy()
    out["panel"] = out.persona.map(panels)
    out["identity"] = identity
    hidden_tag = out.full_text.astype(str).str.contains(
        r"<\s*(?:think|analysis)\b|<\|channel\|>\s*(?:analysis|reasoning)",
        case=False, regex=True)
    if hidden_tag.any():
        raise ValueError("Contoh mengandung tag hidden reasoning")
    out["excerpt_start"] = 0
    excerpts, ends = [], []
    for text in out.full_text.astype(str):
        if len(text) <= 230:
            end = len(text)
        else:
            word_ends = [match.end() for match in re.finditer(r"\S+", text)
                         if match.end() <= 230]
            end = word_ends[-1] if word_ends else 230
        excerpts.append(text[:end])
        ends.append(end)
    out["excerpt_end"] = ends
    out["excerpt"] = excerpts
    out["context_surface"] = "visible_text" if benchmark == "GSM8K" else "final_output"
    out["measured_surface"] = "visible_text" if benchmark == "GSM8K" else "explanation_text"
    if benchmark == "GSM8K":
        out["evidence_note"] = (
            "Triplet purposif; model/soal sama. full_text=visible_text dan surface marker; "
            "tanpa GT/reasoning tersembunyi."
        )
    else:
        out["evidence_note"] = (
            "Triplet purposif; model/soal sama. full_text=final output; marker memakai "
            "explanation_text; tanpa GT/reasoning tersembunyi."
        )
    order = {"N": 0, "E": 1, "I1": 2}
    out["_order"] = out.panel.map(order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if out.panel.tolist() != ["N", "E", "I1"]:
        raise ValueError("Triplet contoh tidak lengkap")
    return out


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coverage_identity_scenario(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out.persona.map(_identity_condition)
    out["identity"] = parsed.map(lambda pair: pair[0])
    out["scenario"] = parsed.map(lambda pair: SCENARIOS[pair[1]])
    return out


@lru_cache(maxsize=1)
def _scan_gsm_records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Satu scan beku untuk denominator, 32 profil, dan kandidat contoh."""
    catalog = pd.read_csv(FILES["GSM8K"]["catalog"])
    markers = catalog.marker_id.tolist()
    count_cols = [f"{marker}__count" for marker in markers]
    present_cols = [f"{marker}__present" for marker in markers]
    usecols = (["model_id", "persona", "question_id", "source_path", "text_available",
                "word_count", "visible_text"] + count_cols + present_cols)
    aggregates, examples = [], []
    wanted = {"neutral", "north_american_explicit", "north_american_implicit_1"}
    for chunk in pd.read_csv(FILES["GSM8K"]["records"], usecols=usecols,
                             chunksize=10_000, low_memory=False):
        available = chunk.text_available.fillna(False).astype(bool)
        numeric = chunk.loc[available, ["model_id", "persona", "word_count"]
                            + count_cols + present_cols].copy()
        for column in ["word_count"] + count_cols + present_cols:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce").fillna(0)
        grouped = numeric.groupby(["model_id", "persona"], as_index=False).sum()
        grouped["n_available"] = (numeric.groupby(["model_id", "persona"]).size()
                                   .reindex(pd.MultiIndex.from_frame(grouped[["model_id", "persona"]]))
                                   .to_numpy())
        aggregates.append(grouped)
        sample = chunk[chunk.persona.isin(wanted) & available].copy()
        if not sample.empty:
            examples.append(sample[["model_id", "persona", "question_id", "source_path",
                                    "visible_text"]])
    combined = pd.concat(aggregates, ignore_index=True)
    sums = combined.groupby(["model_id", "persona"], as_index=False).sum(numeric_only=True)
    denom = sums[["model_id", "persona", "word_count", "n_available"]].rename(
        columns={"word_count": "total_words"})
    profile_rows = []
    for marker in markers:
        part = sums[["model_id", "persona", "n_available",
                     f"{marker}__count", f"{marker}__present"]].copy()
        part["marker_id"] = marker
        part = part.rename(columns={f"{marker}__count": "count_sum",
                                    f"{marker}__present": "n_positive",
                                    "n_available": "n_text"})
        part["estimate"] = np.where(part.n_text.gt(0), 100 * part.n_positive / part.n_text,
                                    np.nan)
        part["prevalence_pct"] = part.estimate
        profile_rows.append(part)
    profiles = pd.concat(profile_rows, ignore_index=True)
    raw_examples = pd.concat(examples, ignore_index=True)
    raw_examples = raw_examples.rename(columns={"visible_text": "full_text"})
    raw_examples["source_path"] = raw_examples.source_path.map(
        lambda value: str((ROOT / value).resolve()) if not Path(str(value)).is_absolute()
        else str(Path(str(value))))
    raw_examples["response_id"] = (raw_examples.model_id + "|" + raw_examples.persona
                                    + "|" + raw_examples.question_id.astype(str))
    raw_examples["pairing_eligible"] = True
    return denom, profiles, raw_examples


def _collapse_gsm_length(source: pd.DataFrame) -> pd.DataFrame:
    rows = source[source.outcome_id.eq("word_count")].copy()
    check_cols = ["n_available", "median", "q1", "q3", "iqr"]
    for _, duplicate in rows.groupby(["model_id", "persona"]):
        if len(duplicate) > 1 and duplicate[check_cols].nunique(dropna=False).max() > 1:
            raise ValueError("Duplikat neutral GSM8K tidak identik")
    rows = rows.drop_duplicates(["model_id", "persona"]).copy()
    rows["estimate"] = rows["median"]
    rows["lower"] = rows["q1"]
    rows["upper"] = rows["q3"]
    rows["interval_kind"] = "IQR"
    return _coverage_identity_scenario(rows)


def _ethics_example_rows() -> pd.DataFrame:
    view = pd.read_csv(FILES["ETHICS"]["analysis_view"],
                       usecols=["model_id", "persona", "question_id", "text_available",
                                "pairing_eligible", "response_key"])
    wanted = {"neutral", "north_american_explicit", "north_american_implicit_1"}
    candidates = view[view.persona.isin(wanted) & view.text_available.astype(bool)
                      & view.pairing_eligible.astype(bool)].copy()
    support = (candidates.groupby(["model_id", "question_id"]).persona
               .agg(lambda values: set(values)).reset_index())
    support = support[support.persona.map(lambda values: wanted.issubset(values))]
    rank = {model: i for i, model in enumerate(MODELS)}
    support["rank"] = support.model_id.map(rank).fillna(len(rank))
    chosen = support.sort_values(["rank", "question_id", "model_id"], kind="mergesort").iloc[0]
    found = []
    with pd.read_csv(FILES["ETHICS"]["records"],
                     usecols=["model", "persona_folder", "file", "question_id",
                              "response_content_full"], chunksize=10_000,
                     low_memory=False) as reader:
        for chunk in reader:
            part = chunk[chunk.model.eq(chosen.model_id) & chunk.question_id.eq(chosen.question_id)
                         & chunk.persona_folder.isin(wanted)].copy()
            if not part.empty:
                found.append(part)
            if sum(len(frame) for frame in found) >= 3:
                break
    rows = pd.concat(found, ignore_index=True).drop_duplicates(
        ["model", "persona_folder", "question_id"])
    rows = rows.rename(columns={"model": "model_id", "persona_folder": "persona",
                                "response_content_full": "full_text"})
    rows["response_id"] = rows.model_id + "|" + rows.persona + "|" + rows.question_id
    rows["pairing_eligible"] = True
    paths = []
    for row in rows.itertuples(index=False):
        if row.persona == "neutral":
            path = (ROOT / "result/benchmarks/ethics/runs/main/v9/Neutral Persona"
                    / row.model_id / row.file)
        else:
            path = (ROOT / "result/benchmarks/ethics/runs/main/v9/Regional Cultural Persona"
                    / row.model_id / row.persona / row.file)
        paths.append(str(path))
    rows["source_path"] = paths
    return rows


def _load_gsm8k() -> dict[str, object]:
    manifest = _read_json(FILES["GSM8K"]["manifest"])
    if manifest.get("response_records") != 134_531 or manifest.get("hidden_reasoning_used") is not False:
        raise ValueError("Manifest GSM8K beku tidak sesuai")
    catalog = pd.read_csv(FILES["GSM8K"]["catalog"], float_precision="round_trip")
    _assert_unique(catalog, ["marker_id"], "catalog GSM8K")
    if len(catalog) != 32:
        raise ValueError("Catalog GSM8K bukan 32 marker")
    coverage = pd.read_csv(FILES["GSM8K"]["coverage"], float_precision="round_trip")
    coverage["available"] = coverage.n_text_available
    coverage["total"] = coverage.target
    coverage["percent"] = 100 * coverage.text_coverage_rate
    coverage = _coverage_identity_scenario(coverage)
    _assert_unique(coverage, ["model_id", "persona"], "coverage GSM8K")
    length = _collapse_gsm_length(pd.read_csv(FILES["GSM8K"]["length"],
                                               float_precision="round_trip"))
    denominators, profiles, example_source = _scan_gsm_records()
    profiles = profiles.merge(coverage[["model_id", "persona", "total", "identity", "scenario"]],
                              on=["model_id", "persona"], validate="many_to_one")
    profiles = profiles.rename(columns={"total": "n_total"})
    profiles["display_unit"] = "percent"
    _assert_unique(profiles, ["model_id", "persona", "marker_id"], "profile GSM8K")
    effects = pd.read_csv(FILES["GSM8K"]["effects"], float_precision="round_trip")
    effects = effects[effects.contrast_id.isin([f"{condition}_vs_neutral"
                                                for condition in CONDITIONS])
                      & effects.outcome_id.isin(catalog.marker_id)].copy()
    overview = adapt_effects(effects, "GSM8K", detail=False)
    detail = adapt_effects(effects, "GSM8K", detail=True)
    phrase = pd.read_csv(FILES["GSM8K"]["terms"], float_precision="round_trip").rename(
        columns={"occurrences": "raw_count"})
    terms = build_terms(phrase, denominators, MODELS, IDENTITIES)
    terms["term_source_kind"] = "frozen_term_group_label"
    terms["placeholder_reason"] = np.where(
        terms.term_available, "", "tidak ada tambahan term teramati; bukan respons missing")
    examples = build_examples(example_source, "GSM8K")
    return _bundle("GSM8K", catalog, coverage, length, profiles, overview, detail,
                   terms, examples)


def _load_ethics() -> dict[str, object]:
    verification = _read_json(FILES["ETHICS"]["verification"])
    provenance = _read_json(FILES["ETHICS"]["statistics_provenance"])
    exclusions = pd.read_csv(FILES["ETHICS"]["pairing_exclusions"])
    if (verification.get("total") != 154_428 or verification.get("n_text") != 126_723
            or provenance.get("main_rows") != 17_472
            or provenance.get("pairing_excluded") != 20 or len(exclusions) != 20
            or not exclusions.pairing_exclusion_reason.eq("source_scenario_mismatch").all()):
        raise ValueError("Policy/pairing ETHICS beku tidak sesuai")
    catalog = pd.read_csv(FILES["ETHICS"]["catalog"], float_precision="round_trip")
    _assert_unique(catalog, ["marker_id"], "catalog ETHICS")
    if len(catalog) != 53:
        raise ValueError("Catalog ETHICS bukan 53 marker")
    coverage = pd.read_csv(FILES["ETHICS"]["coverage"], float_precision="round_trip")
    if not coverage.analysis_policy_version.eq(ETHICS_POLICY).all():
        raise ValueError("Coverage ETHICS memakai policy lain")
    coverage["available"] = coverage.n_text
    coverage["percent"] = coverage.coverage_pct
    coverage = _coverage_identity_scenario(coverage)
    _assert_unique(coverage, ["model_id", "persona"], "coverage ETHICS")
    length = coverage.copy()
    length["estimate"] = length.median_words
    length["lower"] = length.q25_words
    length["upper"] = length.q75_words
    length["interval_kind"] = "IQR"
    profiles = pd.read_csv(FILES["ETHICS"]["profiles"], float_precision="round_trip")
    profiles["estimate"] = profiles.prevalence_pct
    profiles["scenario"] = profiles.condition.map(SCENARIOS)
    profiles["display_unit"] = "percent"
    _assert_unique(profiles, ["model_id", "persona", "marker_id"], "profile ETHICS")
    stats = pd.read_csv(FILES["ETHICS"]["effects"], float_precision="round_trip")
    stats = stats[stats.population.eq("text_all")
                  & stats.contrast_id.isin([f"{condition}_vs_neutral"
                                            for condition in CONDITIONS])
                  & stats.outcome_id.isin(catalog.marker_id)
                  & stats.registered_measurement.eq("density")].copy()
    overview = adapt_effects(stats, "ETHICS", detail=False)
    detail = adapt_effects(stats, "ETHICS", detail=True)
    regional_ids = set(catalog.loc[catalog.registry_scope.eq(
        "regional_cultural_specific"), "marker_id"])
    hits = pd.read_csv(FILES["ETHICS"]["terms"],
                       usecols=["model_id", "persona", "marker_id", "matched_text", "decision"])
    hits = hits[hits.decision.eq("accepted") & hits.marker_id.isin(regional_ids)].copy()
    hits["term"] = hits.matched_text.fillna("").astype(str)
    hits = hits[hits.term.str.strip().ne("")]
    phrase = (hits.groupby(["model_id", "persona", "term"], as_index=False)
              .size().rename(columns={"size": "raw_count"}))
    denominators = coverage[["model_id", "persona", "word_total", "n_text"]].rename(
        columns={"word_total": "total_words", "n_text": "n_available"})
    terms = build_terms(phrase, denominators, MODELS, IDENTITIES)
    terms["term_source_kind"] = "frozen_accepted_matched_text_case_sensitive"
    terms["placeholder_reason"] = np.where(
        terms.term_available, "", "tidak ada tambahan matched_text; bukan respons missing")
    examples = build_examples(_ethics_example_rows(), "ETHICS")
    return _bundle("ETHICS", catalog, coverage, length, profiles, overview, detail,
                   terms, examples)


def _bundle(benchmark: str, catalog: pd.DataFrame, coverage: pd.DataFrame,
            length: pd.DataFrame, profiles: pd.DataFrame, overview: pd.DataFrame,
            detail: pd.DataFrame, terms: pd.DataFrame, examples: pd.DataFrame) -> dict[str, object]:
    marker_labels = {marker: marker.replace("_", " ").title()
                     for marker in catalog.marker_id}
    marker_labels.update(CORE_LABELS)
    marker_labels.update(CATALOG_LABELS)
    notes = {
        "coverage": "Seluruh respons sumber; available berarti visible/explanation text tersedia.",
        "length": "Median kata dan IQR; missing tidak diubah menjadi nol kata.",
        "profile": "Prevalensi respons dengan marker (%); seluruh catalog candidate dipertahankan.",
        "overview": "Paired rank-biserial persona minus Neutral; tanpa CI karena CI sumber milik median delta.",
        "detail": "Median paired difference per 100 kata dengan CI bootstrap 95% marginal dan Holm sumber.",
        "terms": (
            "GSM8K memakai label term/group beku (bukan selalu frasa literal tunggal); "
            "ETHICS memakai accepted matched_text literal case-sensitive. Enam teratas per "
            "identitas, tie alfabetis; placeholder NA berarti tidak ada term teramati tambahan, "
            "bukan respons missing. Inventory kandidat kedua benchmark tidak identik."
        ),
        "examples": "Triplet purposif N/E/I1: model dan soal sama; teks literal, tanpa ground truth.",
    }
    previous_links = {
        "GSM8K": {
            "gallery": str(GSM_VISUAL),
            "source": str(GSM_RUN),
            "documentation": str(ROOT / "docs/linguistic_marker_regional_progress.md"),
        },
        "ETHICS": {
            "notebook": str(ROOT / "notebooks/regional/linguistic marker/ETHICS"
                            / "linguistic_marker_regional_cultural_ethics_v3_candidate.ipynb"),
            "gallery": str(ETHICS_BUNDLE / "evidence/response_cards.html"),
            "source": str(ETHICS_BUNDLE),
        },
    }
    return {
        "benchmark": benchmark,
        "identities": IDENTITIES.copy(),
        "identity_labels": IDENTITY_LABELS.copy(),
        "personas": PERSONAS.copy(),
        "persona_labels": PERSONA_LABELS.copy(),
        "core_markers": CORE_MARKERS.copy(),
        "marker_labels": marker_labels,
        "catalog": catalog,
        "coverage": coverage,
        "length": length,
        "profile": profiles,
        "overview": overview,
        "detail": detail,
        "terms": terms,
        "examples": examples,
        "notes": notes,
        "previous_links": previous_links[benchmark],
    }


@lru_cache(maxsize=2)
def _load_cached(benchmark: str) -> dict[str, object]:
    return _load_gsm8k() if benchmark == "GSM8K" else _load_ethics()


def load(benchmark: str) -> dict[str, object]:
    """Muat adapter benchmark; DataFrame dikopi agar cache tidak termutasi consumer."""
    benchmark = benchmark.upper()
    if benchmark not in FILES:
        raise ValueError(f"Benchmark tidak dikenal: {benchmark}")
    frozen = _load_cached(benchmark)
    return {key: value.copy(deep=True) if isinstance(value, pd.DataFrame)
            else value.copy() if isinstance(value, (dict, list)) else value
            for key, value in frozen.items()}
