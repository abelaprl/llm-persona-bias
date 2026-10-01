import re
import shutil
import subprocess
import unittest
import pandas as pd
from test_religion_lm_v3_foundation import scratch_directory
from src.benchmarks.ethics.build_religion_v3_explorer import build_explorer


class ExplorerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node unavailable')
    def test_local_filters_include_implicit_two_and_render_missing_interval(self):
        with scratch_directory() as directory:
            (directory/'statistics').mkdir();(directory/'evidence').mkdir()
            base=dict(model_id='m',marker_id='theme_x',population='text_all',metric='prevalence_text',
                contrast_kind='vs_neutral',right_persona='neutral',n_pairs=12,effect=5,ci_low=1,ci_high=10,p_holm=.04,
                raw_count_left=8,raw_count_right=3)
            data=[dict(base,left_persona='islam_explicit'),dict(base,left_persona='islam_implicit_2',ci_low=None,ci_high=None),
                dict(base,left_persona='islam_explicit',population='output_all',metric='prevalence_output')]
            pd.DataFrame(data).to_csv(directory/'statistics/paired_contrasts.csv.gz',index=False)
            target=build_explorer(directory)
            script=re.search(r'<script>(.*?)</script>',target.read_text(encoding='utf-8'),re.S).group(1)
            stub='''const elements={};class E{constructor(tag){this.tag=tag;this.children=[];this.value='';this.textContent='';}set id(v){this._id=v;elements[v]=this;}append(v){this.children.push(v);if(this.tag==='select'&&this.value==='')this.value=v.value;}replaceChildren(){this.children=[];}setAttribute(){}addEventListener(){}}
const document={getElementById:k=>elements[k],createElement:t=>new E(t),createElementNS:(n,t)=>new E(t)};
for(const name of ['controls','summary','rows']){const e=new E('div');e.id=name;}
'''
            assertions='''
if(elements.rows.children.length!==2)throw Error('Text contrasts missing');
if(!elements.rows.children[1].children[0].textContent.includes('implicit_2'))throw Error('I2 missing');
if(!elements.rows.children[1].children[3].textContent.includes('NA'))throw Error('Missing CI represented as zero');
elements.population.value='output_all';elements.metric.value='prevalence_output';render();
if(elements.rows.children.length!==1)throw Error('Output filter failed');
elements.metric.value='prevalence_text';render();
if(elements.rows.children.length!==0)throw Error('Unsupported metric retained');
'''
            js=directory/'check.js';js.write_text(stub+script+assertions,encoding='utf-8')
            log=directory/'node.log'
            with log.open('w',encoding='utf-8') as handle:
                result=subprocess.run([shutil.which('node'),str(js)],stdout=handle,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
            self.assertEqual(result.returncode,0,log.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
