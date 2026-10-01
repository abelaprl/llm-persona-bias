"""Display-only adapter Religion GSM8K/ETHICS dari artefak kandidat beku.

Modul ini tidak menjalankan ekstraksi, scoring, bootstrap, atau uji statistik.
Semua estimand dan interval diambil dari tabel sumber yang sudah ada.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
GSM_RUN = (ROOT / "result/GMS8K/reports/Religion_Persona/linguistic_markers"
           / "religion_gsm8k_full_source_aligned_v1_3_20260818")
GSM_VISUAL = (ROOT / "result/GMS8K/reports/Religion_Persona/linguistic_markers/visual"
              / "religion_gsm8k_bab4_visual_v1_3_1_volcano_20260825")
ETH_SOURCE = ROOT / "result/ETHICS/reports/religion_lm_v3_candidate_20260916_r4_overall"
ETH_STYLE = ROOT / "result/ETHICS/reports/religion_ethics_gsm8k_style_20260918_r2"
ETH_EXPRESSION = ROOT / "result/ETHICS/reports/religion_ethics_expression_20260919_r1"
ETH_PRINT = ROOT / "result/ETHICS/reports/religion_ethics_print_20260919_r2"

MODEL_IDS = [
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "deepseek_deepseek-r1-distill-qwen-32b",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
]
IDENTITIES = ["islam", "christian", "jewish", "atheist"]
IDENTITY_LABELS = {
    "islam": "Islam", "christian": "Kristen", "jewish": "Yahudi", "atheist": "Ateis",
}
CONDITIONS = ["explicit", "implicit_1", "implicit_2", "implicit_3"]
SCENARIO = {"neutral": "N", "explicit": "E", "implicit_1": "I1",
            "implicit_2": "I2", "implicit_3": "I3"}
PERSONAS = ["neutral"] + [f"{identity}_{condition}"
                          for identity in IDENTITIES for condition in CONDITIONS]
PERSONA_LABELS = {"neutral": "N"}
for _identity in IDENTITIES:
    for _condition in CONDITIONS:
        PERSONA_LABELS[f"{_identity}_{_condition}"] = (
            f"{IDENTITY_LABELS[_identity]} {SCENARIO[_condition]}"
        )

# Intersection of dimension-specific IDs in both frozen registries.  Identity-specific
# IDs remain in the full catalog/profile; aligned identity is handled by the renderer.
CORE_MARKERS = [
    "religion_generic_reference",
    "religion_deity_invocation",
    "religion_practice_community",
    "religion_formulaic_expression",
    "religion_identity_irrelevance",
    "religion_blessing_invocation",
    "religion_respect_acknowledgement",
]
CORE_LABELS = {
    "religion_generic_reference": "Rujukan agama umum",
    "religion_deity_invocation": "Rujukan Tuhan/deity",
    "religion_practice_community": "Praktik/komunitas agama",
    "religion_formulaic_expression": "Formula agama",
    "religion_identity_irrelevance": "Identitas dinyatakan tidak relevan",
    "religion_blessing_invocation": "Doa/blessing",
    "religion_respect_acknowledgement": "Pengakuan/respect",
}

GSM_FILES = {
    "manifest": GSM_RUN / "manifest.json",
    "visual_manifest": GSM_VISUAL / "manifest.json",
    "coverage": GSM_VISUAL / "response_counts_model_persona.csv",
    "length": GSM_RUN / "condition_descriptives.csv",
    "responses": GSM_RUN / "response_records.csv",
    "catalog": GSM_RUN / "marker_registry.csv",
    "effects": GSM_RUN / "paired_numeric.csv",
    "terms": GSM_VISUAL / "religion_term_prevalence_model_persona.csv",
}
ETH_FILES = {
    "manifest": ETH_SOURCE / "manifest.json",
    "style_manifest": ETH_STYLE / "manifest.json",
    "coverage": ETH_STYLE / "figure_tables/01_coverage_model_persona.csv",
    "profile": ETH_STYLE / "tables/overall_descriptive.csv",
    "effects": ETH_SOURCE / "statistics/paired_contrasts.csv.gz",
    "catalog": ETH_STYLE / "tables/registry.csv",
    "terms": ETH_STYLE / "tables/religious_terms_by_model.csv",
    "examples": ETH_EXPRESSION / "response_cards.json",
    "diversity": ETH_PRINT / "tables/diversity_summary.csv",
}
GSM_EXAMPLE_RAW = {
    "N": ROOT / "result/GMS8K/Gender_Persona/deepseek_deepseek-r1-distill-qwen-32b/neutral/gsm8k_00001.json",
    "E": ROOT / "result/GMS8K/Religion_Persona/deepseek_deepseek-r1-distill-qwen-32b/islam_explicit/gsm8k_00001.json",
    "I1": ROOT / "result/GMS8K/Religion_Persona/deepseek_deepseek-r1-distill-qwen-32b/islam_implicit_1/gsm8k_00001.json",
}


def input_paths() -> list[Path]:
    """Semua file beku yang dibaca adapter, untuk hashing/provenance build."""
    return list(GSM_FILES.values()) + list(ETH_FILES.values()) + list(GSM_EXAMPLE_RAW.values())


def _persona_parts(persona: str) -> tuple[str, str, str]:
    if persona == "neutral":
        return "neutral", "neutral", "N"
    for condition in CONDITIONS:
        suffix = "_" + condition
        if persona.endswith(suffix):
            identity = persona[:-len(suffix)]
            if identity not in IDENTITIES:
                break
            return identity, condition, SCENARIO[condition]
    raise ValueError(f"Persona Religion tidak dikenal: {persona}")


def _add_persona_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parts = out.persona.map(_persona_parts)
    out["identity"] = parts.map(lambda value: value[0])
    out["condition"] = parts.map(lambda value: value[1])
    out["scenario"] = parts.map(lambda value: value[2])
    return out


def _validate_grid(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {label}: {keys}")


def adapt_gsm8k_effects(source: pd.DataFrame, detail: bool) -> pd.DataFrame:
    """Adaptasi efek density GSM8K tanpa memasangkan CI median ke rank-biserial."""
    out = source.copy(deep=True)
    required = {"model_id", "identity", "outcome_id", "contrast_id", "left_condition",
                "right_condition", "left_persona", "right_persona", "rank_biserial",
                "median_paired_delta", "ci_low", "ci_high", "n_pairs", "n_nonzero", "p_holm"}
    missing = required - set(out)
    if missing:
        raise ValueError(f"Kolom efek GSM8K hilang: {sorted(missing)}")
    expected_left = out.identity.astype(str) + "_" + out.left_condition.astype(str)
    direction_ok = (out.right_condition.eq("neutral") & out.right_persona.eq("neutral")
                    & out.left_persona.eq(expected_left)
                    & out.contrast_id.eq(out.left_condition.astype(str) + "_vs_neutral"))
    if not direction_ok.all():
        raise ValueError("Unexpected GSM8K effect direction; wajib persona minus neutral")
    out["marker_id"] = out.outcome_id
    out["row_id"] = out.marker_id
    out["scenario"] = out.left_condition.map(SCENARIO)
    if out.scenario.isna().any():
        raise ValueError("Unknown GSM8K scenario/unit")
    _validate_grid(out, ["model_id", "identity", "scenario", "marker_id"], "GSM8K effect")
    out["changed"] = pd.to_numeric(out.n_nonzero)
    out["corrected_p"] = pd.to_numeric(out.p_holm)
    if detail:
        out["estimate"] = pd.to_numeric(out.median_paired_delta)
        out["lower"] = pd.to_numeric(out.ci_low)
        out["upper"] = pd.to_numeric(out.ci_high)
        out["display_unit"] = "occurrences_per_100_words"
        out["estimand_display"] = "median_paired_density_difference_persona_minus_neutral"
    else:
        out["estimate"] = pd.to_numeric(out.rank_biserial)
        out["lower"] = np.nan
        out["upper"] = np.nan
        out["display_unit"] = "rank_biserial"
        out["estimand_display"] = "matched_rank_biserial_of_density_persona_minus_neutral"
    return out


def adapt_ethics_effects(source: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Adaptasi efek ETHICS pada populasi text_all, dengan gate CI sumber."""
    if metric not in {"prevalence_text", "density"}:
        raise ValueError(f"Unknown ETHICS metric/unit: {metric}")
    out = source.copy(deep=True)
    required = {"model_id", "left_persona", "right_persona", "contrast_kind", "population",
                "marker_id", "metric", "effect", "ci_low", "ci_high", "ci_status", "n_pairs",
                "informative_density_pairs", "discordant_left_only", "discordant_right_only", "p_holm"}
    missing = required - set(out)
    if missing:
        raise ValueError(f"Kolom efek ETHICS hilang: {sorted(missing)}")
    if not (out.metric.eq(metric) & out.population.eq("text_all")
            & out.contrast_kind.eq("vs_neutral")).all():
        raise ValueError("Unexpected ETHICS metric, population, or contrast kind")
    parsed = out.left_persona.map(_persona_parts)
    out["identity"] = parsed.map(lambda value: value[0])
    out["condition"] = parsed.map(lambda value: value[1])
    out["scenario"] = parsed.map(lambda value: value[2])
    if not (out.right_persona.eq("neutral") & out.identity.isin(IDENTITIES)
            & out.condition.isin(CONDITIONS)).all():
        raise ValueError("Unexpected ETHICS effect direction; wajib persona minus neutral")
    out["row_id"] = out.marker_id
    _validate_grid(out, ["model_id", "identity", "scenario", "marker_id"], "ETHICS effect")
    out["estimate"] = pd.to_numeric(out.effect)
    valid_ci = out.ci_status.eq("available")
    out["lower"] = pd.to_numeric(out.ci_low).where(valid_ci)
    out["upper"] = pd.to_numeric(out.ci_high).where(valid_ci)
    if (out.loc[valid_ci, "lower"] > out.loc[valid_ci, "upper"]).any():
        raise ValueError("Invalid ETHICS source CI ordering")
    if metric == "prevalence_text":
        out["changed"] = (pd.to_numeric(out.discordant_left_only)
                          + pd.to_numeric(out.discordant_right_only))
        out["display_unit"] = "percentage_points"
        out["estimand_display"] = "paired_prevalence_difference_pp_persona_minus_neutral"
    else:
        out["changed"] = pd.to_numeric(out.informative_density_pairs)
        out["display_unit"] = "occurrences_per_100_words"
        out["estimand_display"] = "paired_mean_density_difference_persona_minus_neutral"
    out["corrected_p"] = pd.to_numeric(out.p_holm)
    return out


def _gsm8k_profile(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_ids = catalog.marker_id.tolist()
    count_columns = [marker + "__count" for marker in marker_ids]
    columns = ["model_id", "persona", "question_id", "source_path", "visible_text",
               "text_available", "word_count", *count_columns]
    responses = pd.read_csv(GSM_FILES["responses"], usecols=columns, low_memory=False)
    responses["text_available"] = responses.text_available.astype(str).str.lower().eq("true")
    available = responses.loc[responses.text_available].copy()
    if available[count_columns].isna().any().any():
        raise ValueError("Respons GSM8K tersedia memiliki marker count kosong")
    keys = ["model_id", "persona"]
    grouped = available.groupby(keys, sort=False)
    raw = grouped[count_columns].sum().rename(columns=lambda col: col.removesuffix("__count"))
    present = (available[count_columns].gt(0).groupby([available[k] for k in keys]).sum()
               .rename(columns=lambda col: col.removesuffix("__count")))
    denominator = grouped.agg(n_available=("question_id", "size"), word_total=("word_count", "sum"))
    raw_long = raw.stack().rename("raw_count").rename_axis([*keys, "marker_id"]).reset_index()
    present_long = present.stack().rename("n_present").rename_axis([*keys, "marker_id"]).reset_index()
    profile = raw_long.merge(present_long, on=[*keys, "marker_id"], validate="one_to_one")
    profile = profile.merge(denominator.reset_index(), on=keys, validate="many_to_one")
    profile["estimate"] = 100 * profile.n_present / profile.n_available
    profile["display_unit"] = "percent_responses"

    descriptive = pd.read_csv(GSM_FILES["length"])
    descriptive = descriptive.loc[descriptive.outcome_id.isin(marker_ids)].copy()
    check = descriptive.groupby(["model_id", "persona", "outcome_id"])[
        ["n_available", "median", "q1", "q3", "prevalence_nonzero", "median_raw_count"]
    ].nunique(dropna=False)
    if check.gt(1).any().any():
        raise ValueError("Neutral GSM8K descriptives tidak identik antaridentity")
    descriptive = descriptive.drop_duplicates(["model_id", "persona", "outcome_id"])
    profile = profile.merge(
        descriptive.rename(columns={"outcome_id": "marker_id"}),
        on=["model_id", "persona", "marker_id"], how="left", validate="one_to_one",
        suffixes=("", "_source"),
    )
    profile = _add_persona_fields(profile)
    _validate_grid(profile, ["model_id", "persona", "marker_id"], "GSM8K profile")
    return profile, responses


def _gsm8k_length() -> pd.DataFrame:
    source = pd.read_csv(GSM_FILES["length"])
    rows = source.loc[source.outcome_id.eq("word_count")].copy()
    check = rows.groupby(["model_id", "persona"])[["median", "q1", "q3"]].nunique(dropna=False)
    if check.gt(1).any().any():
        raise ValueError("Neutral GSM8K word length tidak identik antaridentity")
    rows = rows.drop_duplicates(["model_id", "persona"])
    rows["estimate"], rows["lower"], rows["upper"] = rows["median"], rows.q1, rows.q3
    rows["interval_kind"] = "IQR"
    rows = _add_persona_fields(rows)
    _validate_grid(rows, ["model_id", "persona"], "GSM8K length")
    return rows


def _select_top_terms(source: pd.DataFrame, count_col: str) -> pd.DataFrame:
    pooled = (source.groupby(["identity", "term"], as_index=False)[count_col].sum()
              .rename(columns={count_col: "pooled_raw_count"}))
    pooled = pooled.sort_values(["identity", "pooled_raw_count", "term"],
                                ascending=[True, False, True])
    pooled["term_order"] = pooled.groupby("identity").cumcount() + 1
    selected = pooled.loc[pooled.term_order.le(6)].copy()
    if selected.groupby("identity").size().reindex(IDENTITIES, fill_value=0).ne(6).any():
        raise ValueError("Tidak tersedia enam term religion untuk setiap identity")
    return selected


def _gsm8k_terms(responses: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(GSM_FILES["terms"])
    source = source.loc[source.condition.eq("explicit") & source.identity.isin(IDENTITIES)].copy()
    selected = _select_top_terms(source, "occurrences")
    denominators = (responses.loc[responses.text_available & responses.persona.str.endswith("_explicit")]
                    .groupby(["model_id", "persona"], as_index=False)
                    .agg(n_available=("question_id", "size"), total_words=("word_count", "sum")))
    denominators = _add_persona_fields(denominators)
    grids = []
    for identity in IDENTITIES:
        terms = selected.loc[selected.identity.eq(identity), ["term", "term_order", "pooled_raw_count"]]
        grid = pd.MultiIndex.from_product([MODEL_IDS, terms.term], names=["model_id", "term"]).to_frame(index=False)
        grid["identity"] = identity
        grid = grid.merge(terms, on="term", validate="many_to_one")
        grids.append(grid)
    grid = pd.concat(grids, ignore_index=True)
    kept = source[["model_id", "identity", "term", "n_available", "occurrences",
                   "responses_with_term", "response_prevalence_pp", "occurrences_per_10k_words"]]
    out = grid.merge(kept, on=["model_id", "identity", "term"], how="left", validate="one_to_one")
    den = denominators[["model_id", "identity", "n_available", "total_words"]]
    out = out.merge(den, on=["model_id", "identity"], how="left", validate="many_to_one",
                    suffixes=("_source", ""))
    if out.total_words.isna().any() or out.n_available.isna().any():
        raise ValueError("Denominator term GSM8K tidak tersedia")
    out["raw_count"] = pd.to_numeric(out.occurrences).fillna(0)
    out["estimate"] = 100 * out.raw_count / out.total_words
    out["row_id"], out["scenario"] = out.term, "E"
    out["display_unit"] = "occurrences_per_100_words"
    return out.sort_values(["identity", "term_order", "model_id"]).reset_index(drop=True)


def _ethics_terms() -> pd.DataFrame:
    source = pd.read_csv(ETH_FILES["terms"])
    if not (source.condition.eq("explicit") & source.identity.isin(IDENTITIES)).all():
        raise ValueError("Unexpected ETHICS term scope")
    selected = _select_top_terms(source, "raw_count")
    den_check = source.groupby(["model_id", "identity"])[["n_text", "word_total"]].nunique(dropna=False)
    if den_check.gt(1).any().any():
        raise ValueError("Denominator term ETHICS tidak konsisten")
    denominators = source.drop_duplicates(["model_id", "identity"])[
        ["model_id", "identity", "n_text", "word_total"]]
    grids = []
    for identity in IDENTITIES:
        terms = selected.loc[selected.identity.eq(identity), ["term", "term_order", "pooled_raw_count"]]
        grid = pd.MultiIndex.from_product([MODEL_IDS, terms.term], names=["model_id", "term"]).to_frame(index=False)
        grid["identity"] = identity
        grids.append(grid.merge(terms, on="term", validate="many_to_one"))
    grid = pd.concat(grids, ignore_index=True)
    values = source[["model_id", "identity", "term", "raw_count", "n_present",
                     "density_sum", "mean_response_density", "prevalence_pct"]]
    out = grid.merge(values, on=["model_id", "identity", "term"], how="left", validate="one_to_one")
    out = out.merge(denominators, on=["model_id", "identity"], validate="many_to_one")
    if out.word_total.isna().any() or out.n_text.isna().any():
        raise ValueError("Denominator term ETHICS tidak tersedia")
    out["raw_count"] = pd.to_numeric(out.raw_count).fillna(0)
    for column in ["n_present", "density_sum"]:
        out[column] = pd.to_numeric(out[column]).fillna(0)
    out["n_available"] = out.n_text
    out["total_words"] = out.word_total
    out["estimate"] = 100 * out.raw_count / out.total_words
    out["row_id"], out["scenario"] = out.term, "E"
    out["display_unit"] = "occurrences_per_100_words"
    return out.sort_values(["identity", "term_order", "model_id"]).reset_index(drop=True)


def _excerpt(text: str, limit: int = 230) -> tuple[str, int, int]:
    full = str(text)
    if len(full) <= limit:
        return full, 0, len(full)
    whitespace = [index for index, char in enumerate(full[:limit + 1]) if char.isspace()]
    end = whitespace[-1] if whitespace else limit
    end = len(full[:end].rstrip())
    return full[:end], 0, end


def _gsm8k_examples(responses: pd.DataFrame) -> pd.DataFrame:
    model_id, question_id = "deepseek_deepseek-r1-distill-qwen-32b", "gsm8k_00001"
    wanted = {"N": "neutral", "E": "islam_explicit", "I1": "islam_implicit_1"}
    chosen = responses.loc[
        responses.text_available & responses.model_id.eq(model_id)
        & responses.question_id.eq(question_id) & responses.persona.isin(wanted.values())
    ].copy()
    if len(chosen) != 3 or set(chosen.persona) != set(wanted.values()):
        raise ValueError("Triplet contoh GSM8K yang diaudit tidak lengkap")
    raw = {panel: json.loads(path.read_text(encoding="utf-8"))
           for panel, path in GSM_EXAMPLE_RAW.items()}
    questions = {panel: record["input"]["question"] for panel, record in raw.items()}
    if len(set(questions.values())) != 1:
        raise ValueError("Triplet contoh GSM8K tidak memuat input.question yang identik")
    question_text = questions["N"]
    question_hash = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
    panel_for = {persona: panel for panel, persona in wanted.items()}
    chosen["panel"] = chosen.persona.map(panel_for)
    chosen = chosen.set_index("panel").loc[["N", "E", "I1"]].reset_index()
    records = []
    for row in chosen.itertuples():
        raw_content = raw[row.panel]["response"]["content"]
        if any(token in raw_content for token in ("<think>", "</think>", "<|channel|>")):
            raise ValueError("Raw response contoh GSM8K memuat channel reasoning tersembunyi")
        raw_response_matches_frozen = str(row.visible_text) == raw_content
        excerpt, start, end = _excerpt(raw_content)
        raw_source_path = str(GSM_EXAMPLE_RAW[row.panel].relative_to(ROOT))
        records.append(dict(
            panel=row.panel,
            model_id=row.model_id, question_id=row.question_id,
            response_id=f"GSM8K|{row.model_id}|{row.persona}|{row.question_id}", identity="islam",
            persona=row.persona, full_text=raw_content, excerpt=excerpt,
            excerpt_start=start, excerpt_end=end,
            question_text=question_text, verified_question_sha256=question_hash,
            problem_view_valid=True,
            raw_response_matches_frozen=raw_response_matches_frozen,
            evidence_note=("Input.question identik (hash diverifikasi); konteks raw, bukan "
                           "triplet frozen terukur. Purposif."),
            source_path=raw_source_path, source_surface="raw_response.content",
            source_raw_path=raw_source_path, frozen_source_path=row.source_path,
        ))
    return pd.DataFrame(records)


def _ethics_examples() -> pd.DataFrame:
    cards = json.loads(ETH_FILES["examples"].read_text(encoding="utf-8"))
    rows = pd.DataFrame(cards)
    wanted = ["neutral", "islam_explicit", "islam_implicit_1"]
    candidates = []
    for card_id, group in rows.loc[rows.card_identity.eq("islam")].groupby("card_id", sort=True):
        if set(wanted).issubset(set(group.persona)):
            candidates.append(card_id)
    if not candidates:
        return pd.DataFrame(columns=["panel", "model_id", "question_id", "response_id", "identity",
                                     "full_text", "excerpt", "excerpt_start", "excerpt_end",
                                     "evidence_note", "source_path", "source_surface"])
    chosen = rows.loc[rows.card_id.eq(sorted(candidates)[0])].set_index("persona").loc[wanted].reset_index()
    records = []
    for row in chosen.itertuples():
        excerpt, start, end = _excerpt(row.final_output)
        records.append(dict(
            panel={"neutral": "N", "islam_explicit": "E", "islam_implicit_1": "I1"}[row.persona],
            model_id=row.model_id, question_id=row.question_id, response_id=row.response_key,
            identity="islam", persona=row.persona, full_text=str(row.final_output), excerpt=excerpt,
            excerpt_start=start, excerpt_end=end,
            evidence_note=("Konteks: raw final_output; dapat memuat Answer. Marker dihitung pada "
                           "linguistic_text lexical, bukan cuplikan ini."),
            source_path=row.source_file, source_surface="raw_final_output_context",
        ))
    return pd.DataFrame(records)


def _catalog_labels(catalog: pd.DataFrame) -> dict[str, str]:
    if "label" in catalog:
        labels = dict(zip(catalog.marker_id, catalog.label))
    else:
        labels = {marker: marker.removeprefix("religion_").replace("_", " ").title()
                  for marker in catalog.marker_id}
    labels.update(CORE_LABELS)
    return labels


def _load_gsm8k() -> dict:
    manifest = json.loads(GSM_FILES["manifest"].read_text(encoding="utf-8"))
    if manifest.get("hidden_reasoning_used") is not False or manifest.get("response_records") != 134531:
        raise ValueError("Unexpected GSM8K frozen manifest")
    catalog = pd.read_csv(GSM_FILES["catalog"])
    if len(catalog) != 30 or catalog.marker_id.duplicated().any():
        raise ValueError("Unexpected GSM8K marker catalog")
    coverage = pd.read_csv(GSM_FILES["coverage"])
    coverage["available"] = coverage.n_text_available
    coverage["total"] = coverage.target
    coverage["percent"] = 100 * coverage.text_coverage_rate
    coverage = _add_persona_fields(coverage)
    _validate_grid(coverage, ["model_id", "persona"], "GSM8K coverage")
    length = _gsm8k_length()
    profile, responses = _gsm8k_profile(catalog)
    effects = pd.read_csv(GSM_FILES["effects"])
    density_ids = set(catalog.loc[catalog.measurement.eq("density"), "marker_id"])
    effects = effects.loc[effects.contrast_id.str.endswith("_vs_neutral")
                          & effects.outcome_id.isin(density_ids)].copy()
    overview = adapt_gsm8k_effects(effects, detail=False)
    detail = adapt_gsm8k_effects(effects, detail=True)
    if (not detail.marker_id.isin(density_ids).all()
            or not detail.display_unit.eq("occurrences_per_100_words").all()):
        raise ValueError("GSM8K detail hanya boleh memuat marker density dalam unit per 100 kata")
    terms = _gsm8k_terms(responses)
    examples = _gsm8k_examples(responses)
    return _package("GSM8K", catalog, coverage, length, profile, overview, detail, terms, examples)


def _load_ethics() -> dict:
    manifest = json.loads(ETH_FILES["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_pending_two_independent_human_reviews":
        raise ValueError("Unexpected ETHICS candidate status")
    catalog = pd.read_csv(ETH_FILES["catalog"])
    if len(catalog) != 59 or catalog.marker_id.duplicated().any():
        raise ValueError("Unexpected ETHICS marker catalog")
    coverage = pd.read_csv(ETH_FILES["coverage"])
    coverage["available"] = coverage.linguistic_available
    coverage["total"] = coverage.total
    coverage["percent"] = coverage.coverage_pct
    coverage = _add_persona_fields(coverage)
    _validate_grid(coverage, ["model_id", "persona"], "ETHICS coverage")
    length = coverage.copy()
    length["estimate"], length["lower"], length["upper"] = (
        length.median_words, length.q25_words, length.q75_words)
    length["interval_kind"] = "IQR"
    profile = pd.read_csv(ETH_FILES["profile"])
    profile["estimate"] = pd.to_numeric(profile.prevalence_text)
    profile["display_unit"] = "percent_responses"
    profile["scenario"] = profile.condition.map(SCENARIO)
    _validate_grid(profile, ["model_id", "persona", "marker_id"], "ETHICS profile")
    usecols = [
        "model_id", "subset", "contrast_kind", "left_persona", "right_persona", "population",
        "marker_id", "family_id", "metric", "n_pairs", "total_pairs", "bootstrap_resamples",
        "status", "effect", "ci_low", "ci_high", "p_raw", "p_holm", "inference_scope_status",
        "raw_count_left", "word_total_left", "positive_responses_left", "raw_count_right",
        "word_total_right", "positive_responses_right", "discordant_left_only",
        "discordant_right_only", "informative_density_pairs", "informative_prevalence_pairs",
        "ci_status", "p_method",
    ]
    effects = pd.read_csv(ETH_FILES["effects"], usecols=usecols)
    base = effects.loc[effects.contrast_kind.eq("vs_neutral") & effects.population.eq("text_all")]
    overview = adapt_ethics_effects(base.loc[base.metric.eq("prevalence_text")], "prevalence_text")
    detail = adapt_ethics_effects(base.loc[base.metric.eq("density")], "density")
    terms = _ethics_terms()
    examples = _ethics_examples()
    return _package("ETHICS", catalog, coverage, length, profile, overview, detail, terms, examples)


def _package(benchmark: str, catalog: pd.DataFrame, coverage: pd.DataFrame,
             length: pd.DataFrame, profile: pd.DataFrame, overview: pd.DataFrame,
             detail: pd.DataFrame, terms: pd.DataFrame, examples: pd.DataFrame) -> dict:
    status = ("candidate v1.3; audit Codex bukan validasi manusia independen" if benchmark == "GSM8K"
              else "candidate v3; dua review manusia independen belum selesai")
    notes = {
        "coverage": ("Denominator adalah seluruh slot sumber per model-persona; available hanya teks visible "
                     "yang dapat diukur. Missing/technical/source gap tidak dijadikan nol marker."),
        "length": ("Titik adalah median jumlah kata dan rentang adalah IQR Q1–Q3, bukan confidence interval. "
                   f"Populasi mengikuti surface beku {benchmark}; status {status}."),
        "profile": ("Estimand deskriptif adalah persentase respons teks tersedia dengan ≥1 hit. Raw occurrence, "
                    "jumlah teks, dan total kata dipertahankan; semua marker catalog disertakan."),
        "overview": (("GSM8K: matched rank-biserial atas selisih density persona−neutral; CI median sumber "
                      "sengaja tidak ditempelkan ke rank.") if benchmark == "GSM8K" else
                     ("ETHICS: selisih prevalensi berpasangan persona−neutral dalam percentage points pada "
                      "populasi text_all; CI hanya jika ci_status=available.")) + f" Status {status}.",
        "detail": (("GSM8K: median paired difference density (occurrence/100 kata) persona−neutral dengan "
                    "CI bootstrap sumber; n_nonzero adalah dukungan perubahan.") if benchmark == "GSM8K" else
                   ("ETHICS: paired mean density difference persona−neutral (occurrence/100 kata), populasi "
                    "text_all; interval marginal bootstrap hanya ditampilkan saat ci_status=available."))
                  + f" Status {status}; p Holm tetap terpisah dari CI.",
        "terms": ("Enam ungkapan religion-specific per identitas dipilih deterministik dari pooled raw hit pada E; "
                  "nilai per model = 100×raw_count/total_words. Pemilihan deskriptif, bukan keluarga inferensi baru."),
        "examples": ("Triplet purposif N/E/I1 memakai soal dan model yang sama, tanpa GT. "
                     "Excerpt adalah substring eksak ≤230 karakter; bukan contoh representatif. "
                     + ("GSM8K memilih triplet dengan jumlah hit tujuh core marker tertinggi; "
                        "full_text berasal dari visible_text." if benchmark == "GSM8K" else
                        "ETHICS memakai card_04 beku: raw final_output sebagai konteks dapat memuat baris Answer. "
                        "Marker sumber diukur pada linguistic_text surface lexical hasil normalisasi, "
                        "bukan pada raw final_output atau cuplikan ini.")),
    }
    previous_links = ({
        "notebook": ROOT / "notebooks/religion/linguistic_marker_religion_gsm8k_bab4_visual_v1_3_1_volcano.ipynb",
        "gallery": ROOT / "notebooks/religion/img",
        "source_bundle": GSM_RUN,
    } if benchmark == "GSM8K" else {
        "notebook": ROOT / "notebooks/religion/linguistic marker/ETHICS/linguistic_marker_religion_ethics_v3_candidate_20260916_r4_overall.ipynb",
        "gallery": ROOT / "notebooks/religion/linguistic marker/ETHICS/figures",
        "source_bundle": ETH_SOURCE,
        "response_cards": ETH_EXPRESSION / "response_cards.html",
    })
    return {
        "benchmark": benchmark,
        "identities": IDENTITIES.copy(),
        "identity_labels": IDENTITY_LABELS.copy(),
        "personas": PERSONAS.copy(),
        "persona_labels": PERSONA_LABELS.copy(),
        "core_markers": CORE_MARKERS.copy(),
        "marker_labels": _catalog_labels(catalog),
        "catalog": catalog,
        "coverage": coverage,
        "length": length,
        "profile": profile,
        "overview": overview,
        "detail": detail,
        "terms": terms,
        "examples": examples,
        "notes": notes,
        "previous_links": previous_links,
    }


def load(benchmark: str) -> dict:
    """Muat satu benchmark dari output beku; benchmark wajib GSM8K atau ETHICS."""
    if benchmark == "GSM8K":
        return _load_gsm8k()
    if benchmark == "ETHICS":
        return _load_ethics()
    raise ValueError(f"Unknown benchmark: {benchmark}")
