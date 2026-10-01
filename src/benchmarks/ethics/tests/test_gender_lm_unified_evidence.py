"""Behavior checks for expression densities and frozen response quotations."""

import importlib
import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE = "src.benchmarks.ethics.gender_lm_unified_evidence"


def adapter():
    assert importlib.util.find_spec(MODULE) is not None, "Unified evidence adapter is not implemented"
    return importlib.import_module(MODULE)


def test_frozen_counts_keep_correct_word_denominators_and_units():
    # Catches wrong per-10k conversion, response denominators, or mean densities.
    module = adapter()
    gsm = module.expression_data("gsm8k")
    row = gsm.loc[gsm.model_id.eq("deepseek_deepseek-r1-distill-qwen-32b")
                  & gsm.panel.eq("Man E") & gsm.term.eq("he")].iloc[0]
    assert (row.raw_count, row.total_words, row.n_available) == (157, 130251, 1305)
    assert abs(row.estimate - 0.12053650259882842) < 1e-12
    ethics = module.expression_data("ethics")
    row = ethics.loc[ethics.model_id.eq("deepseek_deepseek-r1-distill-qwen-32b")
                     & ethics.panel.eq("Man E") & ethics.term.eq("I am a man")].iloc[0]
    assert (row.raw_count, row.total_words, row.n_available) == (1, 145327, 1371)
    assert abs(row.estimate - 100 / 145327) < 1e-12


def test_both_benchmarks_keep_all_models_and_identical_term_rows_between_panels():
    # Catches absent terms/models dropped from one persona's display grid.
    module = adapter()
    for benchmark in ["gsm8k", "ethics"]:
        data = module.expression_data(benchmark)
        assert len(data) == 72
        assert data.groupby(["panel", "term"]).model_id.nunique().eq(6).all()
        assert not data.duplicated(["panel", "term", "model_id"]).any()
        man = data.loc[data.panel.eq("Man E"), "term"].drop_duplicates().tolist()
        woman = data.loc[data.panel.eq("Woman E"), "term"].drop_duplicates().tolist()
        assert man == woman
        assert data.display_unit.eq("occurrences_per_100_words").all()
        assert data.status.eq("candidate").all()
        assert data.estimate.notna().all()
    assert adapter().expression_data("gsm8k").term.drop_duplicates().tolist() == [
        "he", "she", "his", "her", "girls", "boys"]


def test_grid_distinguishes_absent_hits_from_missing_denominators():
    # Catches imputing unavailable groups as zero instead of stopping the export.
    module = adapter()
    counts = pd.DataFrame([
        {"model_id": "a", "persona": "man_explicit_1", "term": "he", "raw_count": 3},
    ])
    denominators = pd.DataFrame([
        {"model_id": "a", "persona": "man_explicit_1", "total_words": 40, "n_available": 2},
        {"model_id": "a", "persona": "woman_explicit_1", "total_words": 80, "n_available": 3},
    ])
    result = module.prepare_expression_grid(counts, denominators, ["he"], models=["a"])
    assert result.raw_count.tolist() == [3, 0]
    assert result.estimate.tolist() == [7.5, 0.0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "denominator"):
        module.prepare_expression_grid(counts, denominators.iloc[:1], ["he"], models=["a"])
    bad = denominators.copy()
    bad.loc[1, "total_words"] = 0
    with unittest.TestCase().assertRaisesRegex(ValueError, "denominator"):
        module.prepare_expression_grid(counts, bad, ["he"], models=["a"])


def test_examples_are_real_complete_triplets_with_exact_bounded_excerpts():
    # Catches unrelated questions, rewritten quotations, and sliced partial words.
    module = adapter()
    for benchmark, question in [("gsm8k", "gsm8k_00004"), ("ethics", "ethics_commonsense_0092")]:
        data = module.example_data(benchmark)
        assert data.panel.tolist() == ["N", "Man E", "Woman E"]
        assert data.question_id.eq(question).all()
        assert data.model_id.eq("deepseek_deepseek-r1-distill-qwen-32b").all()
        assert data.response_id.nunique() == 3
        for row in data.itertuples(index=False):
            assert row.excerpt == row.full_text[row.excerpt_start:row.excerpt_end]
            assert 0 < len(row.excerpt) <= 230
            assert row.excerpt_start == 0 or not (
                row.full_text[row.excerpt_start - 1].isalnum() and row.full_text[row.excerpt_start].isalnum())
            assert row.excerpt_end == len(row.full_text) or not (
                row.full_text[row.excerpt_end - 1].isalnum() and row.full_text[row.excerpt_end].isalnum())
            assert row.evidence_note
    ethics = module.example_data("ethics")
    man = ethics.loc[ethics.panel.eq("Man E")].iloc[0]
    assert man.response_id == "29ca24520cba1c7d5f005e4f"
    assert "gender may perpetuate stereotypes" in man.excerpt
    woman = ethics.loc[ethics.panel.eq("Woman E")].iloc[0]
    assert "because of her gender" in woman.excerpt
    gsm = module.example_data("gsm8k")
    assert "He does this" in gsm.loc[gsm.panel.eq("Man E"), "excerpt"].iloc[0]
    assert "He runs these sessions" in gsm.loc[gsm.panel.eq("Woman E"), "excerpt"].iloc[0]


def test_source_manifest_covers_each_exported_provenance_path():
    # Catches an export that reads untracked frozen data or loses source columns.
    module = adapter()
    paths = {path.resolve() for path in module.input_paths()}
    assert paths and all(path.is_file() for path in paths)
    for benchmark in ["gsm8k", "ethics"]:
        data = module.expression_data(benchmark)
        for column in ["source_count_path", "source_denominator_path"]:
            for value in data[column].unique():
                assert (module.ROOT / value).resolve() in paths
        examples = module.example_data(benchmark)
        for value in examples.source_path.unique():
            assert (module.ROOT / value).resolve() in paths
        assert data.source_count_column.str.len().gt(0).all()
        assert data.denominator_word_column.str.len().gt(0).all()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
