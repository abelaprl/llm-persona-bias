from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.linguistic_markers.config import GENDER_PERSONAS
from src.core.linguistic_markers.csv_loaders import load_csv_records
from src.core.linguistic_markers.loaders import (
    _record_from_path,
    discover_raw_inputs,
)
from src.core.linguistic_markers.registry import MARKERS
from src.core.linguistic_markers.statistics import (
    _bootstrap_ci,
    build_omnibus,
    build_paired_gender_availability,
    build_paired_gender_binary,
    build_paired_gender_numeric,
    build_paired_gender_numeric_correctness_sensitivity,
    build_paired_gender_opening_multinomial,
    build_paired_binary,
    build_paired_numeric,
    build_paired_numeric_correctness_sensitivity,
    build_paired_opening_multinomial,
    holm_adjust,
    paired_stuart_maxwell,
    paired_rank_biserial,
)


class LoaderTests(unittest.TestCase):
    def test_hidden_reasoning_is_never_used_as_content(self):
        payload = {
            "response": {
                "success": True,
                "raw_api_response": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning": "Perhaps secret reasoning"},
                        }
                    ],
                    "usage": {"completion_tokens": 20},
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k_00001.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = _record_from_path(("gsm8k", "model", "neutral", path))
        self.assertEqual(row["response_content"], "")
        self.assertTrue(row["reasoning_present"])
        self.assertTrue(row["response_truncated"])

    def test_ethics_prediction_raw_is_visible_content_fallback(self):
        payload = {
            "prediction": {"raw": "Visible final response."},
            "response": {"success": True, "raw_api_response": {"choices": []}},
            "subset": "justice",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ethics_justice_0001.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            row = _record_from_path(("ethics", "model", "woman_explicit_1", path))
        self.assertEqual(row["response_content"], "Visible final response.")
        self.assertEqual(row["subset"], "justice")
        self.assertEqual(row["identity"], "woman")
        self.assertEqual(row["condition"], "explicit")

    def test_discovery_records_missing_condition_instead_of_fabricating_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "Gender Persona" / "model" / "man_explicit_1"
            available.mkdir(parents=True)
            (available / "ethics_justice_0001.json").write_text("{}", encoding="utf-8")
            with patch(
                "src.core.linguistic_markers.loaders.ETHICS_MAIN_ROOT",
                root,
            ):
                tasks, inventory = discover_raw_inputs("ethics", models=("model",))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(inventory), len(GENDER_PERSONAS))
        statuses = {row["persona"]: row["input_status"] for row in inventory}
        self.assertEqual(statuses["man_explicit_1"], "available_incomplete")
        self.assertEqual(statuses["neutral"], "missing_source")

    def test_gsm_csv_loader_uses_file_stem_not_corrupt_question_id(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "gsm.csv"
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona_folder": "neutral",
                        "file": r"neutral\\gsm8k_00001.json",
                        "question_id": "0",
                        "dataset": "gsm8k_test",
                        "response_content_full": "Visible explanation.",
                        "is_correct": "False",
                    }
                ]
            ).to_csv(csv_path, index=False)
            with patch(
                "src.core.linguistic_markers.csv_loaders.GSM8K_PARSED_CSV",
                csv_path,
            ):
                records, inventory = load_csv_records(
                    "gsm8k", models=("model",), max_records_per_persona=10
                )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question_id"], "gsm8k_00001")
        self.assertEqual(records[0]["input_source"], "parsed_csv")
        self.assertFalse(records[0]["is_correct"])
        self.assertEqual(len(inventory), len(GENDER_PERSONAS))

    def test_ethics_csv_loader_keeps_abstention_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gender = root / "gender" / "model"
            neutral = root / "neutral" / "model"
            gender.mkdir(parents=True)
            neutral.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "persona_folder": "man_explicit_1",
                        "file": "ethics_commonsense_0001.json",
                        "question_id": "bad-id",
                        "dataset": "commonsense",
                        "response_content_full": "I cannot give a judgment.",
                        "answer_status": "abstain",
                        "is_correct": "",
                    }
                ]
            ).to_csv(gender / "results_gender_model.csv", index=False)
            pd.DataFrame(
                columns=["model", "persona_folder", "file", "response_content_full"]
            ).to_csv(neutral / "results_neutral_model.csv", index=False)
            with patch(
                "src.core.linguistic_markers.csv_loaders.ETHICS_PARSED_ROOT",
                root,
            ):
                records, _ = load_csv_records("ethics", models=("model",))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question_id"], "ethics_commonsense_0001")
        self.assertEqual(records[0]["answer_status"], "abstain")
        self.assertEqual(records[0]["response_content"], "I cannot give a judgment.")


def statistics_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for persona_index, persona in enumerate(GENDER_PERSONAS):
        for qid in range(4):
            value = float(qid + persona_index)
            row: dict[str, object] = {
                "model_id": "model",
                "benchmark": "gsm8k",
                "subset": "all",
                "persona": persona,
                "question_id": f"gsm8k_{qid:05d}",
                "word_count": value,
                "avg_sentence_length": value,
                "ttr_300": value / 10,
                "text_available": True,
                "is_correct": True,
                "opening_category": "narrative",
            }
            for marker in MARKERS:
                row[f"{marker.marker_id}__count"] = 0
                row[f"{marker.marker_id}__density"] = 0.0
                row[f"{marker.marker_id}__present"] = 0
            rows.append(row)
    frame = pd.DataFrame(rows)
    # One unavailable neutral response should be pairwise-deleted only for q0.
    mask = (frame["persona"] == "neutral") & (frame["question_id"] == "gsm8k_00000")
    frame.loc[mask, "word_count"] = np.nan
    return frame


class StatisticsTests(unittest.TestCase):
    def test_fast_bootstrap_keeps_degenerate_intervals_exact(self):
        for statistic in ("mean", "median"):
            with self.subTest(statistic=statistic):
                low, high = _bootstrap_ci(
                    np.array([2.5, 2.5, 2.5, 2.5]),
                    statistic,
                    resamples=1_000,
                    seed=1,
                )
                self.assertEqual((low, high), (2.5, 2.5))

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_rank_biserial_direction_follows_left_minus_right(self):
        self.assertEqual(paired_rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)
        self.assertEqual(paired_rank_biserial(np.array([-1.0, -2.0, -3.0])), -1.0)

    def test_planned_contrasts_remain_separate_and_pair_on_question_id(self):
        result = build_paired_numeric(statistics_fixture(), bootstrap_resamples=0)
        selected = result[
            (result["identity"] == "man") & (result["outcome_id"] == "word_count")
        ]
        self.assertEqual(set(selected["contrast_id"]), {
            "explicit_vs_neutral",
            "implicit_1_vs_neutral",
            "implicit_2_vs_neutral",
            "implicit_3_vs_neutral",
            "implicit_1_vs_explicit",
            "implicit_2_vs_explicit",
            "implicit_3_vs_explicit",
        })
        neutral_contrasts = selected[selected["right_condition"] == "neutral"]
        self.assertTrue((neutral_contrasts["n_pairs"] == 3).all())
        explicit_contrasts = selected[selected["right_condition"] == "explicit"]
        self.assertTrue((explicit_contrasts["n_pairs"] == 4).all())

    def test_binary_registry_marker_uses_mcnemar_not_numeric_test(self):
        frame = statistics_fixture()
        frame.loc[
            (frame["persona"] == "man_explicit_1")
            & (frame["question_id"] == "gsm8k_00001"),
            "ai_identity_statement__present",
        ] = 1
        numeric = build_paired_numeric(frame, bootstrap_resamples=0)
        binary = build_paired_binary(frame, bootstrap_resamples=0)
        self.assertNotIn("ai_identity_statement", set(numeric["outcome_id"]))
        selected = binary[
            (binary["identity"] == "man")
            & (binary["outcome_id"] == "ai_identity_statement")
            & (binary["contrast_id"] == "explicit_vs_neutral")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["test_used"], "mcnemar_exact")
        self.assertEqual(selected.iloc[0]["left_only"], 1)

    def test_density_marker_also_gets_supplementary_presence_test(self):
        frame = statistics_fixture()
        frame.loc[
            (frame["persona"] == "man_explicit_1")
            & (frame["question_id"] == "gsm8k_00001"),
            "connective_causal__present",
        ] = 1
        binary = build_paired_binary(frame, bootstrap_resamples=0)
        selected = binary[
            (binary["identity"] == "man")
            & (binary["outcome_id"] == "connective_causal")
            & (binary["contrast_id"] == "explicit_vs_neutral")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["analysis_role"], "supplementary_presence")
        self.assertEqual(selected.iloc[0]["outcome_scale"], "presence")

    def test_correctness_sensitivity_keeps_only_pairs_where_both_are_correct(self):
        frame = statistics_fixture()
        frame.loc[
            (frame["persona"] == "man_explicit_1")
            & (frame["question_id"] == "gsm8k_00001"),
            "is_correct",
        ] = False
        result = build_paired_numeric_correctness_sensitivity(
            frame,
            bootstrap_resamples=0,
        )
        selected = result[
            (result["identity"] == "man")
            & (result["outcome_id"] == "word_count")
            & (result["contrast_id"] == "explicit_vs_neutral")
        ]
        self.assertEqual(len(selected), 1)
        # q0 is unavailable in neutral; q1 is incorrect in explicit.
        self.assertEqual(selected.iloc[0]["n_pairs"], 2)
        self.assertEqual(selected.iloc[0]["pair_filter"], "both_conditions_correct")

    def test_direct_gender_numeric_pairs_man_and_woman_within_condition(self):
        frame = statistics_fixture()
        result = build_paired_gender_numeric(frame, bootstrap_resamples=0)
        selected = result[result["outcome_id"].eq("word_count")]
        self.assertEqual(
            set(selected["contrast_id"]),
            {
                "man_vs_woman_explicit",
                "man_vs_woman_implicit_1",
                "man_vs_woman_implicit_2",
                "man_vs_woman_implicit_3",
            },
        )
        self.assertTrue((selected["n_pairs"] == 4).all())
        self.assertTrue((selected["median_paired_delta"] < 0).all())

    def test_direct_gender_correctness_filter_is_applied_after_pairing(self):
        frame = statistics_fixture()
        frame.loc[
            (frame["persona"] == "woman_explicit_1")
            & (frame["question_id"] == "gsm8k_00001"),
            "is_correct",
        ] = False
        result = build_paired_gender_numeric_correctness_sensitivity(
            frame,
            bootstrap_resamples=0,
        )
        selected = result[
            result["outcome_id"].eq("word_count")
            & result["condition"].eq("explicit")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["n_pairs"], 3)
        self.assertEqual(selected.iloc[0]["pair_filter"], "both_conditions_correct")

    def test_direct_gender_binary_uses_mcnemar(self):
        frame = statistics_fixture()
        frame.loc[
            (frame["persona"] == "man_explicit_1")
            & (frame["question_id"] == "gsm8k_00001"),
            "ai_identity_statement__present",
        ] = 1
        result = build_paired_gender_binary(frame, bootstrap_resamples=0)
        selected = result[
            result["outcome_id"].eq("ai_identity_statement")
            & result["condition"].eq("explicit")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["test_used"], "mcnemar_exact")
        self.assertEqual(selected.iloc[0]["left_only"], 1)

    def test_direct_gender_availability_keeps_source_gaps_in_union(self):
        frame = statistics_fixture()
        frame = frame[
            ~(
                frame["persona"].eq("woman_explicit_1")
                & frame["question_id"].eq("gsm8k_00000")
            )
        ].copy()
        frame["text_status"] = "available"
        result = build_paired_gender_availability(frame, bootstrap_resamples=0)
        selected = result[
            result["outcome_id"].eq("text_available")
            & result["condition"].eq("explicit")
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["n_union_questions"], 4)
        self.assertEqual(selected.iloc[0]["n_right_records"], 3)
        self.assertEqual(selected.iloc[0]["left_only"], 1)

    def test_direct_gender_opening_is_joint_and_condition_specific(self):
        result = build_paired_gender_opening_multinomial(statistics_fixture())
        self.assertEqual(len(result), 4)
        self.assertEqual(
            set(result["condition"]),
            {"explicit", "implicit_1", "implicit_2", "implicit_3"},
        )
        self.assertEqual(
            set(result["test_used"]),
            {"stuart_maxwell_marginal_homogeneity"},
        )

    def test_stuart_maxwell_opening_test_is_joint_and_paired(self):
        test = paired_stuart_maxwell(
            np.array(["narrative", "narrative", "step_first", "step_first"]),
            np.array(["narrative", "step_first", "step_first", "step_first"]),
        )
        self.assertEqual(test["n_categories_observed"], 2)
        self.assertEqual(test["df"], 1)
        self.assertAlmostEqual(test["statistic"], 1.0)
        self.assertAlmostEqual(test["same_category_rate"], 0.75)
        self.assertAlmostEqual(test["marginal_total_variation"], 0.25)

        omnibus_opening = build_paired_opening_multinomial(statistics_fixture())
        self.assertEqual(len(omnibus_opening), 14)
        self.assertEqual(
            set(omnibus_opening["test_used"]),
            {"stuart_maxwell_marginal_homogeneity"},
        )

    def test_binary_registry_marker_uses_cochran_q_for_omnibus(self):
        omnibus = build_omnibus(statistics_fixture())
        selected = omnibus[omnibus["outcome_id"] == "ai_identity_statement"]
        self.assertTrue((selected["test_used"] == "cochran_q").all())


if __name__ == "__main__":
    unittest.main()
