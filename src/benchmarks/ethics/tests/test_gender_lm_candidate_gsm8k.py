"""Behavior checks for the frozen GSM8K presentation adapters."""

import importlib
import importlib.util
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BUNDLE = ROOT / "result/analyses/linguistic_markers/runs/gender_gsm8k_bab4_visual_revision_v3_direct_gender_r2_20260817"
MODULE = "src.benchmarks.ethics.gender_lm_candidate_gsm8k"


def candidate():
    assert importlib.util.find_spec(MODULE) is not None, "GSM8K candidate adapter is not implemented"
    return importlib.import_module(MODULE)


def test_primary_effects_keep_all_indicators_and_do_not_mislabel_median_intervals():
    # Catches silent indicator loss, sign reversal, and attaching delta CIs to ranks.
    module = candidate()
    source = pd.read_csv(BUNDLE / "direct_gender_paired_numeric.csv")
    result = module.prepare_direct_numeric(source)
    assert len(result) == 384
    assert result.outcome_id.nunique() == 16
    assert set(result.panel_scenario) == {"E", "I1", "I2", "I3"}
    first = result.iloc[0]
    assert np.isclose(first.plot_value, -0.026872157159577525)
    assert first.plot_ci_low != first.plot_ci_low
    assert first.plot_ci_high != first.plot_ci_high
    sparse = result.loc[
        result.model_id.eq("deepseek_deepseek-r1-distill-qwen-32b")
        & result.condition.eq("explicit")
        & result.outcome_id.eq("self_reference_first_person")
    ].iloc[0]
    assert (sparse.n_nonzero, sparse.n_pairs, sparse.rank_biserial, sparse.p_holm) == (1, 1298, -1.0, 1.0)
    assert bool(sparse.sparse_support)
    pd.testing.assert_frame_equal(result[source.columns].reset_index(drop=True),
                                  source.loc[source.analysis_role.eq("primary")].reset_index(drop=True))


def test_primary_effects_reject_reversed_or_duplicate_source_contrasts():
    # Catches plots labelled Man-Woman despite a reversed or repeated source row.
    module = candidate()
    source = pd.read_csv(BUNDLE / "direct_gender_paired_numeric.csv", nrows=2)
    wrong = source.copy()
    wrong.loc[0, "left_identity"] = "woman"
    with unittest.TestCase().assertRaisesRegex(ValueError, "direction"):
        module.prepare_direct_numeric(wrong)
    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate"):
        module.prepare_direct_numeric(pd.concat([source, source.iloc[:1]], ignore_index=True))


def test_coverage_keeps_missing_technical_and_source_gaps_distinct():
    # Catches the tempting but false n_missing=0 simplification.
    module = candidate()
    result = module.prepare_coverage(pd.read_csv(BUNDLE / "response_counts_model_persona.csv"))
    assert result.target.sum() == 71226
    assert result.n_text_available.sum() == 70248
    assert result.n_missing_text.sum() == 165
    assert result.n_technical_only.sum() == 802
    assert result.n_source_gap.sum() == 11
    np.testing.assert_allclose(result.text_available_pct + result.text_unavailable_pct, 100)


def test_raw_marker_counts_exclude_missing_and_keep_zero_responses():
    # Catches missing-as-zero, omitted zero counts, and denominator-weighting mistakes.
    module = candidate()
    frame = pd.DataFrame({
        "model_id": ["example"] * 3,
        "persona": ["neutral"] * 3,
        "text_available": [True, True, False],
        "word_count": [50, 150, np.nan],
        "stance_hedge__count": [2, 0, np.nan],
        "stance_hedge__density": [4.0, 0.0, np.nan],
    })
    result = module.aggregate_marker_counts(frame, ["stance_hedge"]).iloc[0]
    assert result.n_available == 2
    assert result.responses_with_marker == 1
    assert result.raw_occurrences == 2
    assert result.total_words == 200
    assert result.mean_density_per_100_words == 2.0
    assert result.pooled_density_per_100_words == 1.0
    assert result.prevalence_pct == 50.0


def test_rank_page_dots_have_exact_source_x_and_separate_model_rows():
    # Catches x jitter, hidden summaries, clipping beyond [-1,1], and <9pt text.
    module = candidate()
    source = pd.read_csv(BUNDLE / "direct_gender_paired_numeric.csv")
    data = module.prepare_direct_numeric(source)
    data = data[data.outcome_id.eq("self_reference_first_person")]
    fig = module.rank_page(data, ["self_reference_first_person"], "Uji tampilan")
    try:
        assert np.isclose(fig.get_figwidth(), 16 / 2.54)
        for ax, scenario in zip(fig.axes, ["E", "I1", "I2", "I3"]):
            offsets = np.concatenate([item.get_offsets() for item in ax.collections])
            panel = data[data.panel_scenario.eq(scenario)]
            np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(panel.rank_biserial))
            assert len(set(offsets[:, 1])) == 6
            assert ax.get_xlim() == (-1.0, 1.0)
        texts = fig.findobj(match=matplotlib.text.Text)
        assert min(text.get_fontsize() for text in texts if text.get_text()) >= 9
    finally:
        plt.close(fig)


def test_lexical_selection_remains_descriptive_and_keeps_all_comparison_groups():
    # Catches top-token filtering that silently drops an identity or proxy scenario.
    module = candidate()
    result = module.select_lexical_examples(pd.read_csv(BUNDLE / "lexical_contrast_token_weights.csv"))
    assert len(result) == 70
    assert result.groupby(["identity", "contrast_id"]).size().eq(5).all()
    assert result.selection_role.eq("descriptive_top5_not_inference").all()
    assert result.groupby(["identity", "contrast_id"]).selection_rank.max().eq(5).all()


def test_actual_panels_fit_print_bounds_with_readable_type():
    # Catches clipped titles, legends and axis labels at the required physical width.
    from src.benchmarks.ethics.gender_lm_presentation_v2_plots import audit_geometry
    module = candidate()
    descriptive = pd.read_csv(module.SOURCE / "condition_descriptives.csv")
    descriptive = descriptive.drop_duplicates(["model_id", "persona", "outcome_id"])
    descriptive["prevalence_pct"] = 100 * descriptive.prevalence_nonzero
    numeric = module.prepare_direct_numeric(pd.read_csv(BUNDLE / "direct_gender_paired_numeric.csv"))
    pronouns = pd.read_csv(BUNDLE / "paired_gender_pronoun_direct.csv")
    pronouns["panel_scenario"] = pronouns.left_persona.map(module.scenario)
    pronouns["plot_value"] = 100 * pronouns.prevalence_delta
    pronouns["plot_ci_low"] = 100 * pronouns.ci_low
    pronouns["plot_ci_high"] = 100 * pronouns.ci_high
    opening = pd.read_csv(BUNDLE / "opening_composition.csv").drop_duplicates(["model_id", "persona", "opening_category"])
    module._style()
    factories = [
        lambda: module._coverage_figure(module.prepare_coverage(pd.read_csv(BUNDLE / "response_counts_model_persona.csv"))),
        lambda: module._verbosity_figure(descriptive),
        lambda: module._profile_figure(descriptive),
        lambda: module.rank_page(numeric, module.PRIMARY[:8], "Perbedaan gender langsung (1/2)"),
        lambda: module.rank_page(numeric, module.PRIMARY[8:], "Perbedaan gender langsung (2/2)"),
        lambda: module._lexical_figure(module.select_lexical_examples(pd.read_csv(BUNDLE / "lexical_contrast_token_weights.csv"))),
        lambda: module._pronoun_profile_figure(pd.read_csv(BUNDLE / "gender_pronoun_class_prevalence.csv")),
        lambda: module._pronoun_direct_figure(pronouns),
        lambda: module._opening_figure(opening),
    ]
    errors = []
    for index, factory in enumerate(factories, start=1):
        figure = factory()
        try:
            geometry = audit_geometry(figure)
            assert geometry["width_cm"] == 16.0
            assert geometry["min_font_pt"] >= 9.0
        except ValueError as error:
            errors.append(f"Panel {index}: {error}")
        finally:
            plt.close(figure)
    assert not errors, "\n".join(errors)


def test_rank_support_text_does_not_cover_any_model_dot():
    # Catches support labels overlapping the following indicator's model points.
    module = candidate()
    data = module.prepare_direct_numeric(pd.read_csv(BUNDLE / "direct_gender_paired_numeric.csv"))
    fig = module.rank_page(data, module.PRIMARY[8:], "Perbedaan gender langsung (2/2)")
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            points = np.concatenate([ax.transData.transform(item.get_offsets()) for item in ax.collections])
            for text in ax.texts:
                bounds = text.get_window_extent(renderer).padded(3)
                assert not any(bounds.contains(x, y) for x, y in points), text.get_text()
    finally:
        plt.close(fig)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
