from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script(
    "build_gsm8k_gender_qwen_truncated_manifest",
    "scripts/build_gsm8k_gender_qwen_truncated_manifest.py",
)
wrapper = load_script(
    "rerun_gsm8k_gender_qwen_truncated_vast",
    "scripts/rerun_gsm8k_gender_qwen_truncated_vast.py",
)


class GenderQwenTruncatedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.classifications = builder.build_manifest()

    def test_exact_317_scope(self) -> None:
        self.assertEqual(len(self.rows), 317)
        self.assertEqual(
            len({str(row["relative_path"]) for row in self.rows}),
            317,
        )
        self.assertEqual(
            self.classifications["selected_length_2048_unparsed"],
            317,
        )

    def test_only_requested_parameters_and_gender_qwen_paths(self) -> None:
        for row in self.rows:
            self.assertEqual(row["dataset"], "GSM8K")
            self.assertEqual(row["group"], "Gender_Persona")
            self.assertEqual(row["model_slug"], builder.MODEL_SLUG)
            self.assertEqual(row["original_finish_reason"], "length")
            self.assertEqual(row["original_max_tokens"], 2048)
            self.assertEqual(row["original_completion_tokens"], 2048)
            self.assertEqual(row["rerun_temperature"], 0)
            self.assertEqual(row["rerun_max_tokens"], 4096)
            self.assertTrue(
                str(row["relative_path"]).startswith(
                    f"Gender_Persona/{builder.MODEL_SLUG}/"
                )
            )
            self.assertNotIn("ground_truth", row)

    def test_parser_recoveries_and_other_failures_are_excluded(self) -> None:
        selected = {str(row["relative_path"]) for row in self.rows}
        self.assertNotIn(
            f"Gender_Persona/{builder.MODEL_SLUG}/man_implicit_1/gsm8k_00518.json",
            selected,
        )
        self.assertNotIn(
            f"Gender_Persona/{builder.MODEL_SLUG}/man_implicit_2/gsm8k_00502.json",
            selected,
        )
        self.assertEqual(self.classifications["recovered_by_current_parser"], 2)
        self.assertEqual(self.classifications["excluded_finish_content_filter"], 3)
        self.assertEqual(self.classifications["excluded_finish_missing"], 1)


class GenderQwenTruncatedWrapperTests(unittest.TestCase):
    def test_fixed_safe_parameters(self) -> None:
        args = wrapper.parse_args([])
        self.assertEqual(args.expected_count, 317)
        self.assertEqual(args.dataset, "gsm8k")
        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.success_root.resolve(), wrapper.SUCCESS_ROOT.resolve())
        self.assertFalse(wrapper.runner.AUTH_REQUIRED)

    def test_scope_override_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            wrapper.parse_args(["--max-tokens", "8192"])


if __name__ == "__main__":
    unittest.main()
