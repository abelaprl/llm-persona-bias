"""Record completed image inspection and independently inspect PDF page metadata."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.benchmarks.ethics.religion_lm_v3_io import write_json,sha256


def record():
    report=ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_candidate_20260916_r4_overall'
    figures=json.loads((report/'figure_manifest.json').read_text(encoding='utf-8'))
    figures+=json.loads((report/'figure_manifest_additional.json').read_text(encoding='utf-8'))
    panels=json.loads((report/'publication_panel_manifest.json').read_text(encoding='utf-8'))
    assert len(figures)==13 and len(panels['panels'])==24
    records=[]
    for figure in figures+panels['panels']:
        path=next(p for p in figure['files'] if p.endswith('.png'))
        records.append(dict(figure_id=figure['figure_id'],path=path,sha256=sha256(report/path),
            inspected_with='view_image high',scope='ethics_all',visible_clipping_observed=False,
            scope_units_and_candidate_caption_checked=True))
    pdf=report/panels['pdf'];content=pdf.read_bytes()
    pages=len(re.findall(rb'/Type\s*/Page(?!s)\b',content))
    assert pages==24
    boxes=re.findall(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]',content)
    assert boxes and all(abs(float(w)-595.44)<.1 and abs(float(h)-841.68)<.1 for w,h in boxes)
    write_json(report/'visual_review.json',dict(status='AI_visual_and_automated_layout_review_complete_not_human_gold',
        reviewed_images=records,image_count=37,pdf_pages=pages,pdf_a4_media_boxes_checked=len(boxes),pdf_sha256=sha256(pdf),
        native_publication_geometry_checks='24 panels assert text bounds and non-overlapping footer blocks during rendering',
        numerical_verification=str(ROOT/'result/benchmarks/ethics/reports/religion_lm_v3_verification_20260916_r5/verification.json'),
        guidance=['Use A4 PDF or individual publication panels for thesis export; six-model forests are overview figures.',
            'Overview implicit forest requires its caption: I1 minus explicit. A4 titles and row labels state conditions explicitly.',
            'A dot without visible interval is not evidence of certainty; CI support and p-Holm are in the accompanying tables.'],
        notebook_link_fix='Windows separators normalized and rebased; reviewed notebook stored under a new filename',
        old_result_files_overwritten=False,script_sha256=sha256(__file__)))
    print('Recorded inspection of 37 PNG files and 24 A4 PDF pages.',flush=True)


if __name__=='__main__':record()
