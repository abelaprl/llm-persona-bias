"""Copy a complete report notebook to a new workspace path, fixing local links."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.benchmarks.ethics.religion_lm_v3_io import write_json


def copy_notebook(report):
    source=report/'religion_ethics_linguistic_markers_v3_presentation_candidate.ipynb'
    notebook=json.loads(source.read_text(encoding='utf-8'))
    destination=ROOT/'notebooks/religion'/('linguistic_marker_religion_ethics_v3_candidate_'+report.name.removeprefix('religion_lm_v3_candidate_')+'.ipynb')
    prefix=os.path.relpath(report,destination.parent).replace('\\','/')+'/'
    for cell in notebook['cells']:
        lines=[]
        for line in cell['source']:
            line=line.replace('](religion_ethics_marker_publication_panels.pdf)',']('+prefix+'religion_ethics_marker_publication_panels.pdf)')
            for folder in ('figures','figure_tables','evidence','registry','statistics','inventory','audit','comparison_with_baseline'):
                line=line.replace(']('+folder+'/',']('+prefix+folder+'/')
            lines.append(line)
        cell['source']=lines
    write_json(destination,notebook)
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True,type=Path)
    copy_notebook(parser.parse_args().report)
