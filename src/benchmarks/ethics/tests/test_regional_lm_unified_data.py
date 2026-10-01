"""Kontrak adapter display-only Regional GSM8K/ETHICS."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.ethics import regional_lm_unified_data as data


class RegionalUnifiedModuleTests(unittest.TestCase):
    def test_effect_adapter_never_attaches_median_ci_to_rank_biserial(self):
        """Menangkap bug pelabelan CI median sebagai CI rank-biserial."""
        self.assertTrue(hasattr(data, "adapt_effects"))
        gsm = pd.DataFrame([
            dict(model_id="m", identity="north_american",
                 contrast_id="explicit_vs_neutral", left_condition="explicit",
                 right_condition="neutral", left_persona="north_american_explicit",
                 right_persona="neutral", outcome_id="culture_general_reference",
                 median_paired_delta=.25, ci_low=-.1, ci_high=.5, n_pairs=100,
                 n_nonzero=12, rank_biserial=.75, p_holm=.04)
        ])
        overview = data.adapt_effects(gsm, "GSM8K", detail=False)
        detail = data.adapt_effects(gsm, "GSM8K", detail=True)
        self.assertEqual(overview.estimate.tolist(), [.75])
        self.assertTrue(overview.lower.isna().all())
        self.assertTrue(overview.upper.isna().all())
        self.assertEqual(overview.estimand_display.tolist(), ["paired_rank_biserial"])
        self.assertEqual(detail.estimate.tolist(), [.25])
        self.assertEqual(detail.lower.tolist(), [-.1])
        self.assertEqual(detail.upper.tolist(), [.5])
        self.assertEqual(detail.estimand_display.tolist(), ["median_paired_difference"])

    def test_ethics_effect_orientation_and_pairing_policy_are_enforced(self):
        """Menangkap pembalikan arah atau pemakaian populasi sensitivitas."""
        self.assertTrue(hasattr(data, "adapt_effects"))
        row = dict(population="text_all", model_id="m", identity="north_american",
                   contrast_id="explicit_vs_neutral", left_condition="explicit",
                   right_condition="neutral", left_persona="north_american_explicit",
                   right_persona="neutral", direction="explicit_minus_neutral",
                   outcome_id="culture_general_reference", registered_measurement="density",
                   outcome_scale="density_per_100_words", effect=.2, ci_low=.1,
                   ci_high=.3, n_pairs=90, n_nonzero=10,
                   effect_size_name="paired_rank_biserial", effect_size=.6,
                   p_holm=.02,
                   analysis_policy_version="regional-explanation-v1-pairing-v2-20260919")
        out = data.adapt_effects(pd.DataFrame([row]), "ETHICS", detail=True)
        self.assertEqual(out.estimate.tolist(), [.2])
        self.assertEqual(out.lower.tolist(), [.1])
        self.assertEqual(out.corrected_p.tolist(), [.02])
        bad = pd.DataFrame([{**row, "direction": "neutral_minus_explicit"}])
        with self.assertRaisesRegex(ValueError, "arah"):
            data.adapt_effects(bad, "ETHICS", detail=True)

    def test_top_terms_are_deterministic_and_zero_fill_only_valid_denominators(self):
        """Menangkap ranking tie yang tidak stabil dan zero-fill tanpa denominator."""
        self.assertTrue(hasattr(data, "build_terms"))
        terms = pd.DataFrame([
            dict(model_id="m1", persona="north_american_explicit", term=term,
                 raw_count=count)
            for term, count in [("z", 3), ("a", 3), ("b", 2), ("c", 2),
                                ("d", 1), ("e", 1), ("f", 1)]
        ])
        denominators = pd.DataFrame([
            dict(model_id="m1", persona="north_american_explicit",
                 total_words=100, n_available=10),
            dict(model_id="m2", persona="north_american_explicit",
                 total_words=200, n_available=20),
        ])
        out = data.build_terms(
            terms, denominators, ["m1", "m2"], ["north_american"], top_n=6
        )
        self.assertEqual(out.loc[out.model_id.eq("m1"), "term"].tolist(),
                         ["a", "z", "b", "c", "d", "e"])
        zero = out[out.model_id.eq("m2")]
        self.assertEqual(zero.raw_count.tolist(), [0] * 6)
        self.assertTrue(np.allclose(zero.estimate, 0))
        self.assertEqual(zero.total_words.unique().tolist(), [200])

    def test_examples_are_literal_same_question_triplet_with_bounded_excerpt(self):
        """Menangkap contoh lintas soal/model atau cuplikan yang bukan substring."""
        self.assertTrue(hasattr(data, "build_examples"))
        rows = []
        for persona, suffix in [("neutral", "N"),
                                ("north_american_explicit", "E"),
                                ("north_american_implicit_1", "I1")]:
            rows.append(dict(model_id="m", persona=persona, question_id="q1",
                             response_id="m|" + persona + "|q1",
                             full_text=(suffix + " ") * 150,
                             source_path="source/" + persona, pairing_eligible=True))
        out = data.build_examples(pd.DataFrame(rows), "GSM8K", "north_american")
        self.assertEqual(out.panel.tolist(), ["N", "E", "I1"])
        self.assertEqual(out.question_id.nunique(), 1)
        self.assertEqual(out.model_id.nunique(), 1)
        self.assertTrue((out.excerpt.str.len() <= 230).all())
        for row in out.itertuples():
            self.assertEqual(row.full_text[row.excerpt_start:row.excerpt_end], row.excerpt)
            self.assertTrue(row.excerpt_end == len(row.full_text)
                            or row.full_text[row.excerpt_end].isspace())
        self.assertEqual(out.context_surface.unique().tolist(), ["visible_text"])
        self.assertEqual(out.measured_surface.unique().tolist(), ["visible_text"])

    def test_frozen_sources_satisfy_complete_cross_benchmark_contract(self):
        """Menangkap source drift, marker hilang, denominator salah, dan pairing rusak."""
        self.assertTrue(all(path.exists() for path in data.input_paths()))
        for benchmark, marker_count in [("GSM8K", 32), ("ETHICS", 53)]:
            bundle = data.load(benchmark)
            self.assertEqual(bundle["identities"], data.IDENTITIES)
            self.assertEqual(bundle["personas"], data.PERSONAS)
            self.assertEqual(bundle["core_markers"], data.CORE_MARKERS)
            self.assertEqual(len(bundle["catalog"]), marker_count)
            self.assertEqual(len(bundle["coverage"]), 102)
            self.assertEqual(len(bundle["length"]), 102)
            self.assertEqual(len(bundle["profile"]), marker_count * 102)
            self.assertEqual(len(bundle["terms"]), 4 * 6 * 6)
            unavailable = bundle["terms"].loc[~bundle["terms"].term_available]
            self.assertTrue(unavailable[["raw_count", "estimate"]].isna().all().all())
            self.assertEqual(len(bundle["examples"]), 3)
            self.assertEqual(bundle["examples"].panel.tolist(), ["N", "E", "I1"])
            self.assertEqual(bundle["examples"].question_id.nunique(), 1)
            self.assertEqual(bundle["examples"].model_id.nunique(), 1)
            self.assertTrue(bundle["examples"].evidence_note.str.len().le(150).all())
            self.assertTrue(set(data.CORE_MARKERS).issubset(bundle["overview"].row_id))
            self.assertTrue(bundle["overview"].lower.isna().all())
            supported = bundle["detail"].n_pairs.gt(0) & bundle["detail"].estimate.notna()
            self.assertTrue(bundle["detail"].loc[supported, ["lower", "upper"]]
                            .notna().all().all())
            self.assertTrue(all(Path(path).exists() for path in bundle["examples"].source_path))
            self.assertTrue(all(Path(path).exists()
                                for path in bundle["previous_links"].values()))


if __name__ == "__main__":
    unittest.main()
