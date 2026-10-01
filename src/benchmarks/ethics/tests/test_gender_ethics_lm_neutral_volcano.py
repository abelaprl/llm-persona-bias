"""Check complete neutral contrasts and display-only transformations."""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics import build_gender_ethics_lm_neutral_volcano as viz


class NeutralVolcanoTest(unittest.TestCase):
    def fixture(self):
        records = []
        for model in viz.contract.MODELS:
            for gender in ("Man", "Woman"):
                for scenario in viz.contract.SCENARIOS:
                    records.append(dict(scope="overall", model_id=model, marker_id="m", measurement="density",
                        left=viz.contract.persona(gender, scenario), right="neutral", effect_direction="left_minus_right",
                        matched_rank_biserial=-.4, adjusted_p=.01, n_changed=50, family="epistemic_stance"))
        registry = pd.DataFrame([dict(marker_id="m", registry_scope="general_ethics", measurement="density")])
        return pd.DataFrame(records), registry

    def test_source_preserved_and_direction_not_reversed(self):
        source, registry = self.fixture()
        rows = viz.prepare(source, registry, "general_ethics")
        pd.testing.assert_frame_equal(rows[source.columns], source)
        self.assertEqual(len(rows), 48)
        self.assertTrue(rows.display_effect.eq(-.4).all())
        self.assertTrue(rows.display_y.eq(2).all())

    def test_na_and_capping_preserve_source(self):
        source, registry = self.fixture()
        source.loc[0, "matched_rank_biserial"] = np.nan
        source.loc[1, "adjusted_p"] = 1e-100
        rows = viz.prepare(source, registry, "general_ethics")
        self.assertFalse(rows.loc[0, "display_available"])
        self.assertTrue(pd.isna(rows.loc[0, "display_effect"]))
        self.assertEqual(rows.loc[1, "display_y"], 50)
        self.assertEqual(rows.loc[1, "adjusted_p"], 1e-100)
        self.assertTrue(rows.loc[1, "display_y_capped"])

    def test_missing_model_scenario_rejected(self):
        source, registry = self.fixture()
        with self.assertRaises(ValueError):
            viz.prepare(source.iloc[1:], registry, "general_ethics")

    def test_eight_panels_and_all_points(self):
        source, registry = self.fixture()
        fig = viz.plot_volcano(viz.prepare(source, registry, "general_ethics"), "Test")
        self.assertEqual(len(fig.axes), 8)
        self.assertEqual(fig.axes[0].get_title(), "Man · E − N")
        self.assertEqual(fig.axes[7].get_title(), "Woman · I3 − N")
        self.assertEqual(sum(len(c.get_offsets()) for ax in fig.axes for c in ax.collections), 48)
        self.assertTrue(all(c.get_alpha() == 1 for ax in fig.axes for c in ax.collections))
        plt.close(fig)

    def test_existing_output_rejected(self):
        with self.assertRaises(FileExistsError):
            viz.build(viz.ROOT)


if __name__ == "__main__":
    unittest.main()
