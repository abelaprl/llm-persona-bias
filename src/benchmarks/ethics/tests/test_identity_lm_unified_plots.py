"""Shared religion/regional geometry and faithful plotted coordinates."""
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.ethics import identity_lm_unified_plots as plots
from src.benchmarks.ethics.gender_lm_reference_data import MODELS


class IdentityPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_regional_catalog_long_names_fit_without_label_overlap(self):
        from src.benchmarks.ethics import regional_lm_unified_data as data
        markers=['culture_meta_framing','culture_relativist_framing',
                 'regional_persona_direct_address','regional_perspective_adoption',
                 'culture_collectivist_communal_orientation','culture_individual_autonomy_orientation']
        bundle=data._bundle('ETHICS',pd.DataFrame({'marker_id':markers}),*[pd.DataFrame() for _ in range(7)])
        rows=pd.DataFrame([dict(panel=p,row_id=r,model_id=m,estimate=0.,display_unit='percent')
                           for p in ['A','B','C','D'] for r in markers for m in MODELS])
        fig=plots.matrix(rows,['A','B','C','D'],markers,bundle['marker_labels'],
                         'Regional ETHICS','CAT07','Prevalence','Note',mode='profile',support=False)
        self.assertEqual(plots.audit_geometry(fig)['min_font_pt'],9)

    def test_missing_term_header_does_not_overlap_long_panel_title(self):
        panels=['Amerika Utara','Asia Timur','Asia Selatan','Afrika Sub-Sahara']
        markers=[f'term{i}' for i in range(6)]
        rows=pd.DataFrame([dict(panel=p,row_id=r,model_id=m,estimate=np.nan,display_unit='count')
                           for p in panels for r in markers for m in MODELS])
        fig=plots.matrix(rows,panels,{p:markers for p in panels},{},'Regional GSM8K',
                         'U06','Unit','Note',mode='positive',support=False)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            missing=next(t for t in ax.texts if t.get_text()=='NA')
            self.assertFalse(ax.title.get_window_extent(renderer).overlaps(missing.get_window_extent(renderer)))

    def test_model_offsets_and_intervals_match_in_both_benchmarks(self):
        panels = ['Islam', 'Christian', 'Jewish', 'Atheist']
        data = pd.DataFrame([dict(panel=p, row_id='hedge', model_id=m, estimate=.2,
                                 lower=-.1, upper=.4, changed=12, display_unit='pp')
                             for p in panels for m in MODELS])
        figs = [plots.matrix(data, panels, ['hedge'], {'hedge':'Hedge'},
                             'Religion '+b, 'U05 · Detail', 'Delta (pp)', 'Note', intervals=True)
                for b in ['GSM8K', 'ETHICS']]
        self.assertEqual(plots.signature(figs[0]), plots.signature(figs[1]))
        self.assertEqual(len(figs[0].axes), 4)
        for ax in figs[0].axes:
            dots = [c for c in ax.collections if (c.get_gid() or '').startswith('model:')]
            self.assertEqual(len(dots), 6)
            self.assertEqual(len(set(float(c.get_offsets()[0, 1]) for c in dots)), 6)
            self.assertTrue(all(float(c.get_offsets()[0, 0]) == .2 for c in dots))
            lines = [c for c in ax.collections if (c.get_gid() or '').startswith('interval:')]
            self.assertEqual(len(lines), 6)
            self.assertEqual(lines[0].get_segments()[0][:, 0].tolist(), [-.1, .4])

    def test_matrix_rejects_mixed_units_or_missing_model_not_silent_zero(self):
        data = pd.DataFrame([dict(panel='E', row_id='x', model_id=m, estimate=0.,
                                 lower=0., upper=0., changed=0, display_unit='pp') for m in MODELS])
        kwargs=dict(panels=['E'], row_ids=['x'], labels={'x':'X'}, context='Test',
                    title='Test', xlabel='Unit', note='Note')
        with self.assertRaisesRegex(ValueError, 'grid'):
            plots.matrix(data.iloc[:-1], **kwargs)
        data.loc[0, 'display_unit']='density'
        with self.assertRaisesRegex(ValueError, 'unit'):
            plots.matrix(data, **kwargs)

    def test_coverage_keeps_seventeen_personas_and_header_separate(self):
        personas=['neutral']+[f'i{x}_{s}' for x in range(4) for s in ['E','I1','I2','I3']]
        rows=pd.DataFrame([dict(model_id=m, persona=p, available=95, total=100, percent=95.)
                           for m in MODELS for p in personas])
        fig=plots.coverage(rows, personas, dict(zip(personas, personas)), 'Religion GSM8K')
        self.assertEqual(fig.axes[0].images[0].get_array().shape, (17,6))
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        top=fig._suptitle.get_window_extent(renderer)
        sub=next(t for t in fig.texts if 'template bersama' in t.get_text()).get_window_extent(renderer)
        self.assertGreater(top.y0-sub.y1, 4*fig.dpi/72)

    def test_length_uses_one_panel_not_one_panel_per_model(self):
        data=pd.DataFrame([dict(model_id=m,persona=p,estimate=10.,lower=5.,upper=20.)
                           for m in MODELS for p in ['N','E']])
        fig=plots.length(data,['N','E'],{'N':'N','E':'E'},'Regional ETHICS','Note')
        self.assertEqual(len(fig.axes),1)
        dots=[c for c in fig.axes[0].collections if (c.get_gid() or '').startswith('model:')]
        self.assertEqual(len(dots),6)
        self.assertTrue(all(len(c.get_offsets())==2 for c in dots))

    def test_overview_has_no_intervals_and_nan_is_not_zero(self):
        data=pd.DataFrame([dict(panel='E',row_id='x',model_id=m,estimate=np.nan if i==0 else 0.,
                                 lower=-1.,upper=1.,changed=0,display_unit='pp')
                           for i,m in enumerate(MODELS)])
        fig=plots.matrix(data,['E'],['x'],{'x':'X'},'Test','U04','Unit','Note',intervals=False)
        all_artists=[c for ax in fig.axes for c in ax.collections]
        self.assertFalse(any((c.get_gid() or '').startswith('interval:') for c in all_artists))
        dots=[c for c in all_artists if (c.get_gid() or '').startswith('model:')]
        self.assertEqual(sum(len(c.get_offsets()) for c in dots),5)
        self.assertTrue(any((c.get_gid() or '').startswith('missing:') for c in all_artists))

    def test_coincident_values_keep_marker_disks_separate_at_print_size(self):
        personas=[f'p{i}' for i in range(17)]
        data=pd.DataFrame([dict(model_id=m,persona=p,estimate=0.,lower=0.,upper=0.)
                           for m in MODELS for p in personas])
        length=plots.length(data,personas,dict(zip(personas,personas)),'Regional ETHICS','Note')
        markers=[f'm{i}' for i in range(7)]
        rows=pd.DataFrame([dict(panel=p,row_id=r,model_id=m,estimate=0.,lower=0.,upper=0.,
                               changed=0,display_unit='pp')
                           for p in ['A','B','C','D'] for r in markers for m in MODELS])
        matrix=plots.matrix(rows,['A','B','C','D'],markers,{},'Religion GSM8K','U04','Unit','Note')
        for fig in [length,matrix]:
            fig.canvas.draw()
            for ax in fig.axes:
                disks=[c for c in ax.collections if (c.get_gid() or '').startswith('model:')]
                centers=np.sort(np.concatenate([ax.transData.transform(c.get_offsets())[:,1] for c in disks]))
                gap=float(np.diff(centers).min())*72/fig.dpi
                diameter=max(float(np.sqrt(c.get_sizes().max())+c.get_linewidths().max()) for c in disks)
                self.assertGreaterEqual(gap,diameter)


if __name__=='__main__':
    unittest.main()
