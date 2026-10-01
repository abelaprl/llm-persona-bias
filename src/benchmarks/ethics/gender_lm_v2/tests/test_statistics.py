import unittest

import numpy as np
import pandas as pd

from src.benchmarks.ethics.gender_lm_v2 import statistics as stats


def long_records(personas, subset="justice"):
    return pd.DataFrame([
        {"model_id": "model", "subset": subset, "question_id": str(i),
         "persona": persona, "value": value,
         "text_available": not pd.isna(value)}
        for persona, values in personas.items() for i, value in enumerate(values)
    ])


class StatisticsTests(unittest.TestCase):
    def test_contrasts_are_predeclared_and_gender_direction_is_positive_woman(self):
        specs = stats.make_contrast_specs()
        self.assertEqual(len({s["contrast_id"] for s in specs}), 27)
        self.assertEqual(sum(s["role"] == "primary" for s in specs), 12)
        self.assertEqual(sum(s["right"] == "neutral" for s in specs), 8)
        direct = [s for s in specs if s["role"] == "primary" and s["right"] != "neutral"]
        self.assertTrue(all(s["left"].startswith("woman_") and s["right"].startswith("man_") for s in direct))

    def test_presence_discordance_sign_and_p_are_paired(self):
        frame = long_records({"woman_explicit_1": [1, 1, 0], "man_explicit_1": [0, 1, 0]})
        pairs = stats.build_pairs(frame, {"left": "woman_explicit_1", "right": "man_explicit_1"})
        result = stats.compare_pairs(pairs, {"measurement": "presence", "bootstrap_resamples": 100})
        self.assertAlmostEqual(result["effect"], 100 / 3)
        self.assertEqual(result["n_changed"], 1)
        self.assertEqual(result["raw_p"], 1)
        reverse = pairs.rename(columns={"left_value": "right_value", "right_value": "left_value"})
        reversed_result = stats.compare_pairs(reverse, {"measurement": "presence", "bootstrap_resamples": 100})
        self.assertAlmostEqual(reversed_result["effect"], -result["effect"])
        self.assertEqual(reversed_result["raw_p"], result["raw_p"])

    def test_pairing_respects_model_subset_and_rejects_duplicates(self):
        frame = long_records({"woman_explicit_1": [1], "man_explicit_1": [0]})
        frame.loc[1, "subset"] = "virtue"
        contrast = {"left": "woman_explicit_1", "right": "man_explicit_1"}
        self.assertEqual(len(stats.build_pairs(frame, contrast)), 0)
        with self.assertRaises(ValueError):
            stats.build_pairs(pd.concat([frame, frame.iloc[:1]]), contrast)

    def test_triplet_identity_and_missing_neutral_does_not_remove_gender_pair(self):
        frame = long_records({"woman_explicit_1": [3, 1], "man_explicit_1": [1, 1], "neutral": [1, 0]})
        result = stats.summarize_neutral_triplets(stats.build_neutral_triplets(frame, "explicit_1"))
        self.assertEqual(result["n_triplets"], 2)
        self.assertEqual(result["d_woman_neutral"], 1.5)
        self.assertEqual(result["d_man_neutral"], 0.5)
        self.assertEqual(result["d_woman_man"], 1.0)
        frame.loc[(frame.persona == "neutral") & (frame.question_id == "1"), "text_available"] = False
        self.assertEqual(len(stats.build_neutral_triplets(frame, "explicit_1")), 1)
        self.assertEqual(len(stats.build_pairs(frame, {"left": "woman_explicit_1", "right": "man_explicit_1"})), 2)

    def test_interaction_requires_four_members_but_no_neutral(self):
        frame = long_records({"woman_implicit_1": [3, 2], "man_implicit_1": [1, 1],
                              "woman_implicit_2": [1, np.nan], "man_implicit_2": [1, 1]})
        result = stats.build_interaction_pairs(frame, "implicit_1", "implicit_2")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.interaction_delta.iloc[0], 2)

    def test_rank_effect_and_sign_effect_can_disagree(self):
        pairs = pd.DataFrame({"subset": ["justice"] * 5, "left_value": [0, 0, 0, 100, 101], "right_value": [1, 1, 1, 0, 0]})
        result = stats.compare_pairs(pairs, {"measurement": "density", "bootstrap_resamples": 100})
        self.assertEqual(result["sign_effect"], -0.2)
        self.assertEqual(result["matched_rank_biserial"], 0.2)
        self.assertGreater(result["effect"], 0)
        self.assertEqual(result["ci_estimand"], "mean_paired_difference")

    def test_holm_joint_neutral_direct_family_and_separate_roles(self):
        frame = pd.DataFrame({"model_id": ["m"] * 4, "scope": ["overall"] * 4,
                              "family": ["f"] * 4, "measurement": ["presence"] * 4,
                              "role": ["primary"] * 3 + ["secondary"], "rule_status": ["candidate"] * 4,
                              "raw_p": [0.01, 0.04, 0.03, 0.02]})
        result = stats.adjust_families(frame)
        np.testing.assert_allclose(result.adjusted_p, [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(result.family_size.tolist(), [3, 3, 3, 1])

    def test_holm_includes_all_measurements_and_missing_hypotheses(self):
        frame = pd.DataFrame({"model_id": ["m"] * 3, "scope": ["overall"] * 3,
                              "family": ["f"] * 3, "measurement": ["count", "density", "presence"],
                              "role": ["primary"] * 3, "raw_p": [0.01, 0.03, np.nan]})
        result = stats.adjust_families(frame)
        np.testing.assert_allclose(result.adjusted_p.iloc[:2], [0.03, 0.06])
        self.assertEqual(result.family_size.tolist(), [3, 3, 3])

    def test_bootstrap_preserves_subset_question_counts(self):
        pairs = pd.DataFrame({"subset": ["justice"] * 2 + ["virtue"] * 8,
                              "left_value": [10] * 2 + [0] * 8, "right_value": [0] * 10})
        result = stats.compare_pairs(pairs, {"measurement": "count", "bootstrap_resamples": 100})
        self.assertEqual(result["effect"], 2)
        self.assertEqual(result["ci_low"], 2)
        self.assertEqual(result["ci_high"], 2)

    def test_availability_permutation_stays_inside_incomplete_question_blocks(self):
        frame = pd.DataFrame({"model_id": ["m"] * 4, "subset": ["justice"] * 4,
                              "question_id": ["0", "0", "1", "1"],
                              "persona": ["neutral", "woman_explicit_1", "neutral", "man_explicit_1"],
                              "text_status": ["substantive_explanation"] * 2 + ["missing"] * 2,
                              "text_available": [True, True, False, False]})
        result = stats._availability_tests(frame, "m", "overall", 99, np.random.default_rng(42))
        self.assertGreater(result[0]["statistic"], 0)
        self.assertEqual(result[0]["raw_p"], 1)
        self.assertEqual(result[1]["raw_p"], 1)

    def test_presence_rejects_nonbinary_even_if_unchanged(self):
        with self.assertRaises(ValueError):
            stats.compare_pairs(pd.DataFrame({"left_value": [2], "right_value": [2]}),
                                {"measurement": "presence", "bootstrap_resamples": 10})

    def test_zero_and_unavailable_are_distinct(self):
        empty = pd.DataFrame(columns=["left_value", "right_value", "subset"])
        unavailable = stats.compare_pairs(empty, {"measurement": "count", "bootstrap_resamples": 100})
        zero = stats.compare_pairs(pd.DataFrame({"left_value": [0], "right_value": [0], "subset": ["justice"]}),
                                   {"measurement": "count", "bootstrap_resamples": 100})
        self.assertTrue(np.isnan(unavailable["effect"]))
        self.assertEqual(zero["effect"], 0)
        self.assertEqual(zero["status"], "no_change")

    def test_run_comparisons_preserves_source_and_masks_missing(self):
        frame = long_records({"woman_explicit_1": [1, 0], "man_explicit_1": [0, 0], "neutral": [0, np.nan]})
        frame["text_status"] = np.where(frame.text_available, "substantive_explanation", "missing")
        frame["word_count"] = np.where(frame.text_available, 10, 0)
        for measurement in ("count", "density", "present"):
            frame["hedge__" + measurement] = frame.value
        before = frame.copy(deep=True)
        result = stats.run_comparisons(frame, [{"marker_id": "hedge", "family": "f", "analysis_role": "inferential", "rule_status": "candidate"}],
                                       bootstrap_resamples=100, availability_permutations=99, include_subsets=False)
        pd.testing.assert_frame_equal(frame, before)
        comparisons = result["comparisons"]
        presence = comparisons[comparisons.measurement == "presence"]
        direct = presence[presence.contrast_id == "woman_explicit_1__minus__man_explicit_1"].iloc[0]
        self.assertEqual(direct.n_pairs, 2)
        self.assertEqual(direct.effect, 50)
        self.assertTrue((comparisons.rule_status == "candidate").all())
        self.assertEqual(set(result), {"comparisons", "neutral_triplet_comparisons", "interaction_comparisons", "pairing_coverage", "availability_tests"})
        self.assertTrue((result["availability_tests"].permutations == 99).all())

    def test_interaction_source_denominator_requires_matching_source_questions(self):
        frame = long_records({"woman_implicit_1": [1], "man_implicit_1": [0],
                              "woman_implicit_2": [1], "man_implicit_2": [0]})
        frame.loc[frame.persona == "man_implicit_1", "question_id"] = "different"
        frame["text_status"] = "substantive_explanation"
        frame["word_count"] = 10
        for measurement in ("count", "density", "present"):
            frame["hedge__" + measurement] = frame.value
        result = stats.run_comparisons(frame, [{"marker_id": "hedge", "family": "f"}], bootstrap_resamples=0,
                                       availability_permutations=0, interaction_permutations=0, include_subsets=False)
        contrast = result["interaction_comparisons"]
        row = contrast[(contrast.contrast_id == "implicit_1__minus__implicit_2") & (contrast.measurement == "presence")].iloc[0]
        self.assertEqual(row.n_left_total, 0)
        self.assertEqual(row.n_right_total, 1)
        self.assertEqual(row.n_pairs, 0)

    def test_model_partitioning_does_not_change_randomized_statistics(self):
        frame = long_records({"woman_explicit_1": [1, 5, 11], "man_explicit_1": [3, 2, 1], "neutral": [0, np.nan, 4]})
        frame["text_status"] = np.where(frame.text_available, "substantive_explanation", "missing")
        frame["word_count"] = 10
        for measurement in ("count", "density", "present"):
            frame["hedge__" + measurement] = frame.value if measurement != "present" else np.where(frame.text_available, frame.value.gt(0).astype(float), np.nan)
        other = frame.assign(model_id="other")
        registry = [{"marker_id": "hedge", "family": "f"}]
        options = dict(bootstrap_resamples=19, availability_permutations=0, interaction_permutations=19, include_subsets=False)
        whole = stats.run_comparisons(pd.concat([frame, other], ignore_index=True), registry, **options)
        self.assertTrue(whole["comparisons"].model_seed.map(lambda value: isinstance(value, str) and value.startswith("0x")).all())
        parts = [stats.run_comparisons(part, registry, **options) for part in (frame, other)]
        for name in whole:
            pd.testing.assert_frame_equal(whole[name].reset_index(drop=True),
                                          pd.concat([part[name] for part in parts], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
