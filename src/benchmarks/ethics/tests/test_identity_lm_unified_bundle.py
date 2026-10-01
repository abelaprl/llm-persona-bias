"""Artifact boundaries for the shared regional/religion presentation."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.benchmarks.ethics.build_identity_lm_unified import axis_label, build
from src.benchmarks.ethics.identity_lm_unified_bundle import make_notebook, write_gallery


class IdentityBundleTests(unittest.TestCase):
    def test_unverified_pairs_are_withheld_not_plotted_as_zero(self):
        from src.benchmarks.ethics.build_identity_lm_unified import withhold_effects
        source=pd.DataFrame({'estimate':[.5], 'lower':[.2], 'upper':[.8],
                             'rank_biserial':[.5], 'n_pairs':[100]})
        display=withhold_effects(source)
        self.assertTrue(display[['estimate','lower','upper']].isna().all().all())
        self.assertEqual(display.source_estimate.tolist(),[.5])
        self.assertEqual(display.rank_biserial.tolist(),[.5])
        self.assertEqual(source.estimate.tolist(),[.5])
        self.assertEqual(display.display_status.tolist(),['withheld_question_pairing_not_validated'])

    def test_unknown_effect_unit_is_not_silently_labelled_as_density(self):
        rows=pd.DataFrame({'display_unit':['words'], 'estimand_display':['mean_difference']})
        with self.assertRaises(ValueError):
            axis_label(rows)

    def test_existing_destination_is_refused_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            marker=path/'keep.txt'
            marker.write_text('original',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build('religion',path)
            self.assertEqual(marker.read_text(encoding='utf-8'),'original')

    def test_notebook_binds_given_output_and_keeps_all_sections(self):
        record=dict(id='gsm8k_U01',benchmark='GSM8K',pair_id='U01',section='main',
                    title='Coverage',caption='Candidate only',png='figures/a.png',svg='figures/a.svg',csv='tables/a.csv')
        nb=make_notebook('religion','GSM8K',[record],Path('D:/a candidate'),{})
        self.assertEqual(nb.metadata['figure_ids'],['gsm8k_U01'])
        setup=next(c.source for c in nb.cells if c.cell_type=='code')
        self.assertIn("D:/a candidate",setup)
        for c in nb.cells:
            if c.cell_type=='code':
                compile(c.source,'<notebook>','exec')

    def test_gallery_places_both_benchmarks_under_each_analysis(self):
        records=[dict(id=b+'_U01',benchmark=b,pair_id='U01',section='main',title='Cakupan',
                      caption='Bukan validasi',png=b+'.png',svg=b+'.svg',csv=b+'.csv') for b in ['GSM8K','ETHICS']]
        with tempfile.TemporaryDirectory() as folder:
            write_gallery(Path(folder),'religion',records,[])
            html=(Path(folder)/'index.html').read_text(encoding='utf-8')
            self.assertLess(html.index('GSM8K.png'),html.index('ETHICS.png'))
            self.assertEqual(html.count('<section id="U01">'),1)
            self.assertIn('tables/gsm8k_catalog.csv',html)


if __name__=='__main__':
    unittest.main()
