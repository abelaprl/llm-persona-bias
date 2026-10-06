"""Presentation invariants independent of extraction or ground truth."""
import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.MAIN_PARSING.ETHICS.complete_lm_notebook_presentation import (
    ROOT, explorer, heterogeneity, normalize, summarize,
)


class PresentationTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(
            population="all_visible", model_id="test_model", subset="commonsense",
            condition="explicit", contrast_id="a_vs_b_explicit", identity_contrast="a_vs_b",
            left_persona="a_explicit", right_persona="b_explicit", outcome_id="test_presence",
            outcome_tier="identity_specific", outcome_scale="binary", effect=-.2,
            effect_size=.4, effect_size_name="matched_odds_ratio", n_pairs=50, n_nonzero=12,
            ci_low=-.3, ci_high=-.1, p_raw=.01, p_holm=.04,
            multiplicity_family="direct_24", test_used="mcnemar_exact",
        )])

    def test_presence_direction_is_not_odds_ratio_sign(self):
        d = normalize(self.fixture(), ROOT / "result/example.csv", "direct")
        self.assertEqual(d.iloc[0].direction, "right_higher")
        self.assertEqual(d.iloc[0].signed_effect, -.2)
        self.assertEqual(d.iloc[0].effect_size, .4)
        self.assertEqual(d.iloc[0].ci_estimand, "prevalence_delta")

    def test_zero_pairs_never_count_as_testable(self):
        raw = self.fixture()
        raw.loc[0, ["n_pairs", "n_nonzero"]] = 0
        raw.loc[0, ["effect", "effect_size", "ci_low", "ci_high"]] = np.nan
        raw.loc[0, "p_holm"] = 1.0
        d = normalize(raw, ROOT / "result/example.csv", "direct")
        s = summarize(d).iloc[0]
        self.assertEqual(s.not_testable, 1)
        self.assertEqual(s.not_significant, 0)
        self.assertEqual(s.unavailable, 1)

    def test_numeric_ci_stays_on_median_difference(self):
        raw = self.fixture()
        raw.loc[0, "outcome_scale"] = "density"
        raw.loc[0, "effect_size_name"] = "paired_rank_biserial"
        raw.loc[0, "effect_size"] = -.8
        d = normalize(raw, ROOT / "result/example.csv", "direct")
        self.assertEqual(d.iloc[0].ci_estimand, "median_paired_delta")
        self.assertEqual(d.iloc[0].estimate, -.2)
        self.assertEqual(d.iloc[0].signed_effect, -.8)
        self.assertEqual(d.iloc[0].ci_low, -.3)

    def test_heterogeneity_does_not_treat_zero_or_na_as_reversal(self):
        d = normalize(self.fixture(), ROOT / "result/example.csv", "direct")
        zero = d.copy()
        zero["subset"] = "justice"
        zero["direction"] = "zero"
        zero["signed_effect"] = 0.0
        self.assertFalse(heterogeneity(pd.concat([d, zero])).iloc[0].opposite_directions)
        positive = zero.copy()
        positive["direction"] = "left_higher"
        positive["signed_effect"] = .1
        positive["decision"] = "not_significant"
        self.assertTrue(heterogeneity(pd.concat([d, positive])).iloc[0].opposite_directions)

    def test_explorer_preserves_small_p_values_and_numeric_precision(self):
        raw = self.fixture()
        raw.loc[0, "p_holm"] = 2.123456789123456e-12
        d = normalize(raw, ROOT / "result/example.csv", "direct")
        with patch('pathlib.Path.write_text') as write:
            explorer(d, ROOT / 'result/test-not-written', "Test")
            page = write.call_args.args[0]
        payload = json.loads(page.split('<script id="payload" type="application/json">')[1].split('</script>')[0])
        self.assertEqual(payload['data'][0][payload['columns'].index('p_holm')], raw.loc[0, 'p_holm'])
        self.assertEqual(payload['data'][0][payload['columns'].index('signed_effect')], -.2)


if __name__ == "__main__":
    unittest.main()
