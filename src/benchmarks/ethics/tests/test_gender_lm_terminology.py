"""Presentation terminology must never corrupt model names or machine keys."""
import unittest
from src.benchmarks.ethics.gender_lm_terminology import display_text


class TerminologyTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(display_text("Man − Woman"), "Laki-laki − Perempuan")
        self.assertEqual(display_text("DeepSeek R1 32B"), "DeepSeek R1-Distill 32B")
        self.assertEqual(display_text("Qwen3 Next 80B"), "Qwen3 Next 80B-A3B")

    def test_idempotent(self):
        for value in ["Qwen3 Next 80B", "DeepSeek R1 32B", "penanda bahasa etika", "Man − Woman"]:
            self.assertEqual(display_text(display_text(value)), display_text(value))

    def test_machine_keys(self):
        for value in ["man_explicit_1", "woman_implicit_2", "explicit_vs_neutral", "Man_minus_Woman.csv"]:
            self.assertEqual(display_text(value), value)


if __name__ == "__main__":
    unittest.main()
