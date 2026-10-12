"""Independent read-only checks for a completed regional presentation run."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest['source_hashes'].items():
        assert digest(path)==expected,('source changed',path)
    for path,expected in manifest['output_hashes'].items():
        assert digest(run/path)==expected,('output changed',path)
    figures=pd.read_csv(run/'figure_index.csv')
    assert len(figures)==29 and figures.folder.eq('bab6').sum()==10
    assert figures.vector.sum()==28
    for row in figures.itertuples():
        png=run/row.folder/(row.stem+'.png')
        with Image.open(png) as image:
            image.load()
            if row.vector:
                assert abs(image.info['dpi'][0]-300)<.1
                assert abs(image.width/300*2.54-14)<.02
        if row.vector:
            pdf=png.with_suffix('.pdf').read_bytes()
            box=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]',pdf)
            assert box, png
            assert abs(float(box[1])*2.54/72-14)<1e-6
            assert b'/Subtype /Image' not in pdf
            assert row.min_font_pt>=8
    cells=pd.read_csv(run/'tables/heatmap_cells.csv')
    summary=pd.read_csv(run/'tables/condition_summary_frozen.csv').set_index(['model','persona','subset'])
    effects=pd.read_csv(run/'tables/paired_effects_frozen.csv').set_index(['outcome','model','persona','subset'])
    for row in cells.itertuples():
        if row.kind=='summary':
            original=summary.loc[(row.model,row.persona,row.subset)]
            np.testing.assert_allclose(row.value_pct,original[row.metric]*100,equal_nan=True)
            np.testing.assert_allclose(row.coverage_pct,100*original.committed/original.total)
        else:
            original=effects.loc[(row.metric,row.model,row.persona,row.subset)]
            np.testing.assert_allclose(row.value_pct,original.risk_difference_pp,equal_nan=True)
            np.testing.assert_allclose(row.p_holm,original.holm_adjusted_p_value,equal_nan=True)
            assert row.significant==original.significant_holm_0_05
    accuracy=cells[(cells.kind=='summary')&(cells.metric=='accuracy_committed')]
    assert len(accuracy)==216 and len(accuracy.drop_duplicates(['model','persona','subset']))==204
    assert len(cells[cells.kind=='effect'])==960
    reasons=pd.read_csv(run/'tables/reason_cells.csv')
    np.testing.assert_allclose(reasons.percent,100*reasons['count']/reasons.denominator.replace(0,np.nan),equal_nan=True)
    distributions=pd.read_csv(run/'tables/distribution_cells.csv')
    np.testing.assert_allclose(distributions.percent,100*distributions['count']/distributions.denominator.replace(0,np.nan),equal_nan=True)
    np.testing.assert_allclose(distributions.groupby(['figure','model','category']).percent.sum(),100)
    assert pd.read_csv(run/'tables/layout_audit.csv').fits.all()
    log=(run/'proof/regional_14cm.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull \\[hv]box|LaTeX Error|Package \w+ Error',log)
    text=subprocess.run(['pdftotext','-layout',str(run/'proof/regional_14cm.pdf'),'-'],capture_output=True,encoding='utf-8',check=True).stdout
    pages=[p for p in text.split('\f') if p.strip()]
    assert len(pages)==len(figures)
    assert '…' not in text and '??' not in text
    for page,row in zip(pages,figures.itertuples()):
        # Captions must finish on the same page; whitespace and line wrapping vary.
        normalize=lambda s: re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',s).casefold())
        assert normalize(row.caption) in normalize(page),(row.stem,'complete caption missing')
    mapping=pd.read_csv(run/'mapping_integrasi.csv')
    assert mapping.old_path.dropna().nunique()==10
    return {'figures':len(figures),'main_figures':10,'pdf_vectors':28,'proof_pages':len(pages),
            'accuracy_cells_unique':204,'paired_effect_cells':960,
            'protected_sources':len(manifest['source_hashes']),'source_and_output_hashes_match':True,
            'all_numerical_checks_passed':True,'all_new_fonts_at_least_8pt':True,
            'physical_print_checked':False,'thesis_modified':False}


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(validate(sys.argv[1]),indent=2))
