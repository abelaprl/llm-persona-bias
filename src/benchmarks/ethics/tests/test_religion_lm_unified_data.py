"""Kontrak adapter Religion lintas GSM8K dan ETHICS pada artefak beku."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.ethics.religion_lm_unified_data import (
    CORE_MARKERS,
    _excerpt,
    adapt_ethics_effects,
    adapt_gsm8k_effects,
    input_paths,
    load,
)


EXPECTED_MODELS = {
    "deepseek_deepseek-r1-distill-qwen-32b",
    "meta-llama_llama-3.1-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "mistralai_ministral-8b-2512",
    "openai_gpt-oss-20b",
    "qwen_qwen3-next-80b-a3b-thinking",
}
EXPECTED_PERSONAS = {
    "neutral",
    *{
        f"{identity}_{condition}"
        for identity in ("islam", "christian", "jewish", "atheist")
        for condition in ("explicit", "implicit_1", "implicit_2", "implicit_3")
    },
}


class FrozenReligionAdapterTests(unittest.TestCase):
    def test_input_paths_are_existing_frozen_sources_and_exclude_secrets(self):
        paths = input_paths()
        self.assertGreaterEqual(len(paths), 10)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertFalse(any("model_keys.json" in path.as_posix() for path in paths))

    def assert_common_contract(self, result, benchmark):
        self.assertEqual(result["benchmark"], benchmark)
        self.assertEqual(result["identities"], ["islam", "christian", "jewish", "atheist"])
        self.assertEqual(set(result["personas"]), EXPECTED_PERSONAS)
        self.assertEqual(result["core_markers"], CORE_MARKERS)
        self.assertEqual(len(result["coverage"]), 102)
        self.assertEqual(len(result["length"]), 102)
        self.assertEqual(set(result["coverage"].model_id), EXPECTED_MODELS)
        self.assertEqual(set(result["coverage"].persona), EXPECTED_PERSONAS)
        self.assertTrue(result["coverage"].percent.between(0, 100).all())
        self.assertTrue((result["length"].interval_kind == "IQR").all())
        self.assertTrue((result["length"].lower <= result["length"].estimate).all())
        self.assertTrue((result["length"].estimate <= result["length"].upper).all())
        self.assertFalse(result["profile"].duplicated(["model_id", "persona", "marker_id"]).any())
        self.assertEqual(len(result["terms"]), 4 * 6 * 6)
        self.assertFalse(result["terms"].duplicated(["model_id", "identity", "row_id"]).any())
        self.assertEqual(set(result["terms"].scenario), {"E"})
        self.assertEqual(result["examples"].panel.tolist(), ["N", "E", "I1"])
        self.assertEqual(result["examples"].identity.unique().tolist(), ["islam"])
        self.assertEqual(result["examples"].model_id.nunique(), 1)
        self.assertEqual(result["examples"].question_id.nunique(), 1)
        for row in result["examples"].itertuples():
            self.assertLessEqual(len(row.excerpt), 230)
            self.assertEqual(row.full_text[row.excerpt_start:row.excerpt_end], row.excerpt)
            if row.excerpt_end < len(row.full_text):
                self.assertTrue(row.full_text[row.excerpt_end].isspace())
                self.assertFalse(row.excerpt[-1].isspace())
        self.assertEqual(set(result["notes"]), {
            "coverage", "length", "profile", "overview", "detail", "terms", "examples"
        })
        self.assertTrue(result["previous_links"])
        self.assertTrue(all(path.exists() for path in result["previous_links"].values()))

    def test_gsm8k_load_preserves_all_catalog_markers_and_true_estimands(self):
        result = load("GSM8K")
        self.assert_common_contract(result, "GSM8K")
        self.assertEqual(len(result["catalog"]), 30)
        self.assertEqual(len(result["profile"]), 6 * 17 * 30)
        self.assertEqual(set(result["profile"].scenario), {"N", "E", "I1", "I2", "I3"})
        self.assertEqual(set(result["overview"].display_unit), {"rank_biserial"})
        self.assertTrue(result["overview"].lower.isna().all())
        self.assertTrue(result["overview"].upper.isna().all())
        self.assertEqual(set(result["detail"].display_unit), {"occurrences_per_100_words"})
        self.assertTrue(set(CORE_MARKERS).issubset(set(result["detail"].marker_id)))
        self.assertTrue((result["overview"].changed == result["overview"].n_nonzero).all())
        self.assertEqual(set(result["examples"].source_surface), {"raw_response.content"})
        density_ids = set(result["catalog"].loc[result["catalog"].measurement.eq("density"), "marker_id"])
        self.assertTrue(set(result["detail"].marker_id).issubset(density_ids))
        source_prevalence = result["profile"].prevalence_nonzero.notna()
        np.testing.assert_allclose(result["profile"].loc[source_prevalence, "estimate"],
                                   100 * result["profile"].loc[source_prevalence, "prevalence_nonzero"])
        # Some frozen descriptives omit binary outcomes; the saved counts still
        # define prevalence for every marker without converting missing text to zero.
        np.testing.assert_allclose(result["profile"].estimate,
                                   100 * result["profile"].n_present / result["profile"].n_available)
        examples = result["examples"]
        self.assertEqual(examples.model_id.unique().tolist(),
                         ["deepseek_deepseek-r1-distill-qwen-32b"])
        self.assertEqual(examples.question_id.unique().tolist(), ["gsm8k_00001"])
        self.assertEqual(examples.question_text.nunique(), 1)
        self.assertEqual(examples.verified_question_sha256.nunique(), 1)
        self.assertTrue(examples.source_raw_path.map(lambda value: Path(value).is_file()).all())
        self.assertTrue(examples.problem_view_valid.all())
        self.assertEqual(examples.raw_response_matches_frozen.tolist(), [False, True, True])
        for row in examples.itertuples():
            raw = json.loads(Path(row.source_raw_path).read_text(encoding="utf-8"))
            self.assertEqual(row.full_text, raw["response"]["content"])
        self.assertTrue(examples.evidence_note.str.len().le(150).all())

    def test_ethics_load_keeps_candidate_ci_gate_and_source_units(self):
        result = load("ETHICS")
        self.assert_common_contract(result, "ETHICS")
        self.assertEqual(len(result["catalog"]), 59)
        self.assertEqual(len(result["profile"]), 6 * 17 * 59)
        self.assertEqual(set(result["overview"].display_unit), {"percentage_points"})
        self.assertEqual(set(result["detail"].display_unit), {"occurrences_per_100_words"})
        invalid = result["detail"].ci_status.ne("available")
        self.assertTrue(result["detail"].loc[invalid, ["lower", "upper"]].isna().all().all())
        valid = result["detail"].ci_status.eq("available")
        self.assertTrue((result["detail"].loc[valid, "lower"] <=
                         result["detail"].loc[valid, "upper"]).all())
        self.assertTrue((result["detail"].changed ==
                         result["detail"].informative_density_pairs).all())
        np.testing.assert_allclose(result["profile"].estimate, result["profile"].prevalence_text)
        self.assertGreater(result["profile"].prevalence_text.max(), 1)
        self.assertEqual(set(result["examples"].source_surface), {"raw_final_output_context"})
        self.assertTrue(result["examples"].evidence_note.str.contains("linguistic_text").all())

    def test_excerpt_is_an_exact_prefix_cut_at_whitespace(self):
        text = "one two three four"
        excerpt, start, end = _excerpt(text, limit=12)
        self.assertEqual((excerpt, start, end), ("one two", 0, 7))
        self.assertEqual(text[start:end], excerpt)
        self.assertTrue(text[end].isspace())

    def test_effect_adapters_reject_wrong_direction_duplicates_and_unknown_metric(self):
        gsm = pd.DataFrame([dict(
            model_id="m", identity="islam", outcome_id=CORE_MARKERS[0],
            contrast_id="explicit_vs_neutral", left_condition="explicit",
            right_condition="neutral", left_persona="islam_explicit",
            right_persona="neutral", rank_biserial=.5, median_paired_delta=.2,
            ci_low=.1, ci_high=.3, n_pairs=10, n_nonzero=4, p_holm=.8,
        )])
        out = adapt_gsm8k_effects(gsm, detail=False)
        self.assertEqual(out.estimate.tolist(), [.5])
        wrong = gsm.assign(right_persona="christian_explicit")
        with self.assertRaisesRegex(ValueError, "direction"):
            adapt_gsm8k_effects(wrong, detail=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapt_gsm8k_effects(pd.concat([gsm, gsm], ignore_index=True), detail=False)

        ethics = pd.DataFrame([dict(
            model_id="m", left_persona="islam_explicit", right_persona="neutral",
            contrast_kind="vs_neutral", population="text_all", marker_id=CORE_MARKERS[0],
            metric="density", effect=.2, ci_low=.1, ci_high=.3, ci_status="available",
            n_pairs=10, informative_density_pairs=4,
            discordant_left_only=1, discordant_right_only=2, p_holm=.8,
        )])
        density = adapt_ethics_effects(ethics, metric="density")
        self.assertEqual(density.estimate.tolist(), [.2])
        with self.assertRaisesRegex(ValueError, "metric"):
            adapt_ethics_effects(ethics, metric="unknown")

    def test_unknown_benchmark_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "benchmark"):
            load("MMLU")


if __name__ == "__main__":
    unittest.main()
