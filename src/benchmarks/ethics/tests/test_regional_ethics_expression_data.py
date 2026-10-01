import unittest
import pandas as pd
from src.benchmarks.ethics.regional_ethics_expression_data import diversity_rows, diversity_tables, profile_table


def fixture():
    return pd.DataFrame([dict(response_key='m|neutral|q', model_id='m', persona='neutral', identity='neutral', condition='neutral',
                             explanation_available=True, word_count=100, analysis_answer_status='abstain', a__count=5, b__count=2,
                             a__density=5., b__density=2.),
                         dict(response_key='m|neutral|q2', model_id='m', persona='neutral', identity='neutral', condition='neutral',
                             explanation_available=False, word_count=1, analysis_answer_status='committed', a__count=0, b__count=0,
                             a__density=0., b__density=0.)])


class ExpressionTests(unittest.TestCase):
    def test_categories_not_occurrences(self):
        rows = diversity_rows(fixture(), ['a', 'b'])
        self.assertEqual(rows.iloc[0].distinct_markers, 2)
        self.assertEqual(rows.iloc[0].raw_marker_count, 7)
        self.assertEqual(rows.iloc[0].marker_density, 7)
        self.assertTrue(pd.isna(rows.iloc[1].distinct_markers))

    def test_profile_abstain_included(self):
        table = profile_table(fixture(), ['a', 'b'])
        self.assertEqual(table.n_text.tolist(), [1, 1])
        self.assertEqual(table.prevalence_pct.tolist(), [100, 100])

    def test_invalid_count_rejected(self):
        for value in [-1, .5]:
            data = fixture()
            data['a__count'] = data['a__count'].astype(float)
            data.loc[0, 'a__count'] = value
            with self.assertRaises(ValueError):
                diversity_rows(data, ['a', 'b'])

    def test_ecdf_complete_and_length_empty_bins(self):
        tables = diversity_tables(diversity_rows(fixture(), ['a', 'b']), n_markers=2)
        dist = tables['diversity_distribution']
        self.assertEqual(dist.distinct_markers.tolist(), [0, 1, 2])
        self.assertEqual(dist.n.sum(), 1)
        self.assertEqual(dist.ecdf.iloc[-1], 1)
        self.assertEqual(len(tables['diversity_by_length']), 5)
