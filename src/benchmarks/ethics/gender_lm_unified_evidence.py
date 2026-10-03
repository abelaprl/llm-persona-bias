"""Read-only expression and quotation adapters for the unified candidate figures.

Counts and denominators come from frozen exports; no marker is redetected and
no response is selected or filtered using correctness or an extracted answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .gender_lm_reference_data import MODELS, ROOT


TABLES = ROOT / "notebooks/gender/linguistic marker/candidate/tables"
ETHICS_SOURCE = ROOT / "result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260916_r3"
GSM_SOURCE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_full_source_aligned_v1_2_1_20260817"
GSM_BUNDLE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817"
PANELS = {"man_explicit_1": "Man E", "woman_explicit_1": "Woman E"}
EXAMPLE_PERSONAS = ["neutral", *PANELS]
EXAMPLE_MODEL = "deepseek_deepseek-r1-distill-qwen-32b"
SELECTION = "union_top3_per_persona_fill_to6_pooled_hits_descending_then_exact_term"


def input_paths() -> list[Path]:
    """List every data input used by these adapters for provenance hashing."""
    return [GSM_BUNDLE / "gendered_term_model_persona.csv",
            GSM_BUNDLE / "gendered_lexicon_metadata.json",
            GSM_SOURCE / "response_records.csv",
            TABLES / "ethics_expression_forms.csv",
            ETHICS_SOURCE / "response_markers.csv.gz",
            TABLES / "ethics_example_responses.csv",
            TABLES / "ethics_example_spans.csv"]


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _selected_terms(counts: pd.DataFrame) -> list[str]:
    per_persona = counts.groupby(["persona", "term"], as_index=False).raw_count.sum()
    chosen = set()
    for persona in PANELS:
        ranked = per_persona.loc[per_persona.persona.eq(persona)].sort_values(
            ["raw_count", "term"], ascending=[False, True])
        chosen.update(ranked.head(3).term)
    total = per_persona.groupby("term", as_index=False).raw_count.sum().sort_values(
        ["raw_count", "term"], ascending=[False, True])
    for term in total.term:
        if len(chosen) >= 6:
            break
        chosen.add(term)
    return [term for term in total.term if term in chosen]


def prepare_expression_grid(counts, denominators, terms, models=MODELS):
    """Fill absent hits only after verifying all requested group denominators."""
    keys = ["model_id", "persona"]
    if denominators.duplicated(keys).any():
        raise ValueError("Duplicate expression denominator")
    if counts.duplicated(keys + ["term"]).any():
        raise ValueError("Duplicate expression count")
    if counts.raw_count.isna().any() or counts.raw_count.lt(0).any():
        raise ValueError("Invalid expression count")
    grid = pd.DataFrame([
        {"model_id": model, "persona": persona, "term": term, "term_order": order}
        for persona in PANELS for order, term in enumerate(terms) for model in models
    ])
    grid = grid.merge(denominators, on=keys, how="left", validate="many_to_one")
    if (grid[["total_words", "n_available"]].isna().any().any()
            or grid.total_words.le(0).any() or grid.n_available.le(0).any()):
        raise ValueError("Missing or nonpositive expression denominator")
    grid = grid.merge(counts, on=keys + ["term"], how="left", validate="one_to_one")
    grid["raw_count"] = grid.raw_count.fillna(0).astype(int)
    grid["estimate"] = 100.0 * grid.raw_count / grid.total_words
    grid["panel"] = grid.persona.map(PANELS)
    grid["display_unit"] = "occurrences_per_100_words"
    grid["estimand"] = "pooled_occurrences_divided_by_total_words"
    grid["selection_rule"] = SELECTION
    grid["status"] = "candidate"
    return grid


def expression_data(benchmark: str) -> pd.DataFrame:
    """Return six exact forms and the complete six-model, two-persona grid."""
    benchmark = benchmark.lower()
    if benchmark == "gsm8k":
        count_path = GSM_BUNDLE / "gendered_term_model_persona.csv"
        with (GSM_BUNDLE / "gendered_lexicon_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["text_field"] != "visible_text" or metadata["hidden_reasoning_used"]:
            raise ValueError("Unexpected GSM8K expression surface")
        source = pd.read_csv(count_path)
        source = source.loc[source.persona.isin(PANELS) & source.model_id.isin(MODELS)].copy()
        grouped = source.groupby(["model_id", "persona"])
        if grouped[["total_words", "n_responses_available"]].nunique().ne(1).any().any():
            raise ValueError("Inconsistent GSM8K expression denominator")
        counts = source[["model_id", "persona", "term", "occurrences"]].rename(
            columns={"occurrences": "raw_count"})
        denominators = source[["model_id", "persona", "total_words", "n_responses_available"]].drop_duplicates().rename(
            columns={"n_responses_available": "n_available"})
        terms = _selected_terms(counts)
        if len(counts.loc[counts.term.isin(terms)]) != len(MODELS) * len(PANELS) * len(terms):
            raise ValueError("Incomplete GSM8K expression count grid")
        source_column, word_column = "occurrences", "total_words"
        denominator_path = count_path
        surface, kind = "visible_text", "closed_lexicon_word"
    elif benchmark == "ethics":
        count_path = TABLES / "ethics_expression_forms.csv"
        source = pd.read_csv(count_path, keep_default_na=False)
        source = source.loc[source.persona.isin(PANELS) & source.model_id.isin(MODELS)]
        counts = source.groupby(["model_id", "persona", "text"], as_index=False).span_hits.sum().rename(
            columns={"text": "term", "span_hits": "raw_count"})
        denominator_path = ETHICS_SOURCE / "response_markers.csv.gz"
        records = pd.read_csv(denominator_path, usecols=[
            "response_id", "model_id", "persona", "text_available", "word_count"])
        if records.response_id.duplicated().any():
            raise ValueError("Duplicate ETHICS response denominator")
        available = records.loc[records.text_available.eq(True) & records.persona.isin(PANELS)
                                & records.model_id.isin(MODELS)]
        if available.word_count.isna().any() or available.word_count.le(0).any():
            raise ValueError("Invalid available ETHICS word denominator")
        denominators = available.groupby(["model_id", "persona"], as_index=False).agg(
            total_words=("word_count", "sum"), n_available=("response_id", "size"))
        terms = _selected_terms(counts)
        source_column, word_column = "span_hits", "word_count"
        surface, kind = "explanation_text", "registered_candidate_span"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    result = prepare_expression_grid(counts, denominators, terms)
    result["benchmark"] = benchmark
    result["source_count_path"] = _relative(count_path)
    result["source_denominator_path"] = _relative(denominator_path)
    result["source_count_column"] = source_column
    result["denominator_word_column"] = word_column
    result["surface_name"] = surface
    result["expression_kind"] = kind
    return result


def _excerpt(text: str, anchor: int, limit: int = 230):
    """Return an exact substring with whitespace-delimited word boundaries."""
    words = list(re.finditer(r"\S+", text))
    start = next((match.start() for match in words if match.start() >= max(0, anchor - 75)), 0)
    end = max((match.end() for match in words if start < match.end() <= start + limit), default=start)
    if not start <= anchor < end:
        raise ValueError("Evidence anchor does not fit its excerpt")
    return text[start:end], start, end


def example_data(benchmark: str) -> pd.DataFrame:
    """Return a purposive same-model, same-question N/Man E/Woman E triplet."""
    benchmark = benchmark.lower()
    if benchmark == "ethics":
        path = TABLES / "ethics_example_responses.csv"
        records = pd.read_csv(path, keep_default_na=False)
        records = records.loc[records.example_id.eq("contoh_1")].copy()
        spans = pd.read_csv(TABLES / "ethics_example_spans.csv", keep_default_na=False)
        spans = spans.loc[spans.example_id.eq("contoh_1")]
        surface = "explanation_text"
        question = "ethics_commonsense_0092"
    elif benchmark == "gsm8k":
        path = GSM_SOURCE / "response_records.csv"
        columns = ["model_id", "persona", "question_id", "visible_text", "text_available",
                   "source_path", "source_row", "raw_source_file"]
        question = "gsm8k_00004"
        records = pd.read_csv(path, usecols=columns, keep_default_na=False)
        records = records.loc[records.model_id.eq(EXAMPLE_MODEL) & records.question_id.eq(question)
                              & records.persona.isin(EXAMPLE_PERSONAS)].copy()
        if not records.text_available.eq(True).all():
            raise ValueError("Unavailable GSM8K example text")
        records["response_id"] = records.model_id + "|" + records.persona + "|" + records.question_id
        records = records.rename(columns={"source_path": "upstream_source_path"})
        spans = None
        surface = "visible_text"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if (len(records) != 3 or set(records.persona) != set(EXAMPLE_PERSONAS)
            or not records.model_id.eq(EXAMPLE_MODEL).all() or not records.question_id.eq(question).all()):
        raise ValueError("Incomplete or mismatched example triplet")
    rows = []
    for persona in EXAMPLE_PERSONAS:
        record = records.loc[records.persona.eq(persona)].iloc[0].to_dict()
        text = record.pop(surface)
        anchor = 0
        note = "Cuplikan purposif; tidak ada span gender candidate pada penjelasan ini."
        if benchmark == "ethics":
            hits = spans.loc[spans.response_id.eq(record["response_id"])].sort_values("start")
            for hit in hits.itertuples(index=False):
                if hit.surface_name != surface or text[int(hit.start):int(hit.end)] != hit.text:
                    raise ValueError("Stored example span does not match frozen text")
            if len(hits):
                hit = hits.iloc[0]
                anchor = int(hit.start)
                note = f"{hit.rule_id}; span {hit.start}:{hit.end}; candidate, referen belum diaudit."
                record.update(evidence_start=int(hit.start), evidence_end=int(hit.end), evidence_text=hit.text)
        else:
            phrase = {"neutral": "To determine", "man_explicit_1": "He does this",
                      "woman_explicit_1": "He runs these sessions"}[persona]
            anchor = text.index(phrase)
            note = ("Cuplikan purposif; 'He' muncul pada kalimat tentang James. Ini konteks tokoh soal."
                    if persona != "neutral" else "Cuplikan purposif; penjelasan netral untuk soal James yang sama.")
            record.update(evidence_start=anchor, evidence_end=anchor + len(phrase), evidence_text=phrase)
            record["response_id_kind"] = "composite_model_persona_question_key"
        if benchmark == "gsm8k":
            # Fixed prose boundaries for this inspected example avoid cutting math.
            first = ("To determine" if persona == "neutral" else "James runs **3 sprints** each session.")
            last = {"neutral": "follow these steps:",
                    "man_explicit_1": "He does this **3 times a week**.",
                    "woman_explicit_1": "He runs these sessions **3 times a week**."}[persona]
            start = text.index(first)
            end = text.index(last, start) + len(last)
            quote = text[start:end]
            if len(quote) > 230:
                raise ValueError("Fixed prose quotation exceeds the print limit")
        else:
            quote, start, end = _excerpt(text, anchor)
        record.update(panel="N" if persona == "neutral" else PANELS[persona], full_text=text,
                      excerpt=quote, excerpt_start=start, excerpt_end=end, evidence_note=note,
                      source_path=_relative(path), surface_name=surface, benchmark=benchmark,
                      status="candidate", selection_role="purposive_not_representative")
        rows.append(record)
    return pd.DataFrame(rows)


def evidence_notes(benchmark: str) -> dict[str, str]:
    """Supply short Indonesian captions describing selection and limitations."""
    benchmark = benchmark.lower()
    if benchmark not in {"gsm8k", "ethics"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    form = ("Kata dari kamus tertutup pada visible_text GSM8K" if benchmark == "gsm8k"
            else "Frasa exact-span terdaftar pada explanation_text ETHICS, dengan kapitalisasi asli")
    return {
        "expressions": form + ". Gabungan tiga bentuk dengan hit terbanyak per persona E dilengkapi hingga enam "
            "menurut total hit gabungan; tie diurutkan alfabetis. Frekuensi = 100 × jumlah hit / total kata "
            "seluruh teks tersedia, tanpa filter jawaban benar; bukan rerata density per respons. "
            "Hit dapat merujuk tokoh soal. Kamus dan permukaan teks antardataset berbeda; status candidate.",
        "examples": "Triplet purposif DeepSeek R1 pada satu soal yang sama untuk N, Man E, dan Woman E. "
            "Kutipan merupakan cuplikan teks sumber; teks utuh dan posisi karakter tersimpan pada CSV. "
            "Contoh tidak mewakili prevalensi atau membuktikan bias terhadap pengguna; status candidate.",
    }
