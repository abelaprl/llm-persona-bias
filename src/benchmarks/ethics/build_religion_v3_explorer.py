"""Local, dependency-free explorer for every prespecified whole-ETHICS contrast."""
import json
import pandas as pd
from src.benchmarks.ethics.religion_lm_v3_io import sha256,write_json


def build_explorer(output):
    frame = pd.read_csv(output/'statistics/paired_contrasts.csv.gz')
    categories = ['model_id','marker_id','population','metric']
    fields = categories+['contrast_kind','left_persona','right_persona','n_pairs','effect','ci_low','ci_high','p_holm','raw_count_left','raw_count_right']
    data = frame[fields].to_json(orient='values',double_precision=6)
    options = {name:sorted(frame[name].unique()) for name in categories}
    template = '''<!doctype html><html lang="id"><meta charset="utf-8">
<title>Religion ETHICS — seluruh kontras marker</title>
<style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1500px;padding:15px;color:#18313b}select{max-width:340px;padding:8px;margin:5px}label{display:inline-block;margin-right:14px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}thead{position:sticky;top:0;background:#edf3f4}svg{vertical-align:middle}small{color:#566}#notice{background:#fff1cf;padding:10px}h1{font-size:26px}</style>
<h1>Linguistic marker Religion — keseluruhan ETHICS</h1>
<p id="notice">Kandidat; precision/recall manusia belum tersedia. Semua 52 kontras per model tersedia, termasuk implicit 1, 2, dan 3 secara terpisah. Tema leksikal bukan endorsement atau penilaian kualitas moral.</p>
<div id="controls"></div><p id="summary"></p>
<table><thead><tr><th>Kontras (kiri − kanan)</th><th>n pasangan</th><th>Efek dan CI 95%</th><th>Nilai efek [CI]</th><th>p Holm</th><th>Occurrence kiri / kanan</th></tr></thead><tbody id="rows"></tbody></table>
<p><small>text_all: kedua respons memiliki teks; output_all: kedua final tersedia, termasuk label-only. both_committed_text memakai status parser kandidat upstream. equal_subset_text memberi bobot setara pada empat subset; tanpa dukungan semua subset hasil NA. no_quality_flag_text mengecualikan flag kualitas yang ditetapkan. Density adalah selisih rata-rata occurrence per 100 kata per respons. Prevalence memakai poin persentase. Koreksi Holm: model × population × metric × jenis kontras × keluarga marker. Uji prevalence tanpa bobot: McNemar eksak; density dan prevalence equal-subset: paired sign-flip dengan asumsi exchangeability dalam pertanyaan. CI bootstrap bersifat perkiraan marginal 95%, bukan interval simultan; dukungan kurang atau bootstrap degenerat ditandai NA. Nol tidak sama dengan NA. Jumlah occurrence adalah total mentah pada pasangan terpilih, bukan estimasi berbobot. Analisis utama tidak mensyaratkan committed.</small></p>
<script>const data=__DATA__, options=__OPTIONS__;
const names={model_id:'Model',marker_id:'Marker',population:'Populasi',metric:'Metrik'};
Object.entries(options).forEach(([name,values])=>{const label=document.createElement('label');label.textContent=names[name]+' ';const select=document.createElement('select');select.id=name;values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;select.append(o)});label.append(select);document.getElementById('controls').append(label);select.addEventListener('change',render)});
document.getElementById('population').value='text_all';document.getElementById('metric').value='prevalence_text';
function fmt(x){return x===null?'NA':Number(x).toFixed(3)}
function fmtP(x){return x===null?'NA':x<.001?'<0.001':Number(x).toFixed(3)}
function render(){const choices=Object.keys(options).map(k=>document.getElementById(k).value);const rows=data.filter(r=>choices.every((v,i)=>v===r[i]));const finite=rows.flatMap(r=>r.slice(8,11)).filter(v=>v!==null);const limit=choices[3]==='density'?Math.max(1,...finite.map(Math.abs))*1.05:100;const x=v=>180+v/limit*175;
document.getElementById('summary').textContent=rows.length+' kontras; skala efek −'+limit.toFixed(2)+' hingga +'+limit.toFixed(2)+'. '+(rows.length?'':'Kombinasi metric/population ini tidak didefinisikan; pilih prevalence_output untuk output_all.');
const body=document.getElementById('rows');body.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');const values=[r[5]+' − '+r[6],r[7],null,fmt(r[8])+' ['+fmt(r[9])+', '+fmt(r[10])+']',fmtP(r[11]),r[12]+' / '+r[13]];values.forEach((v,i)=>{const td=document.createElement('td');if(i!==2)td.textContent=v;else{const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('width',360);svg.setAttribute('height',26);function line(a,b,color,width){const e=document.createElementNS(ns,'line');e.setAttribute('x1',x(a));e.setAttribute('x2',x(b));e.setAttribute('y1',13);e.setAttribute('y2',13);e.setAttribute('stroke',color);e.setAttribute('stroke-width',width);svg.append(e)}const zero=document.createElementNS(ns,'line');zero.setAttribute('x1',180);zero.setAttribute('x2',180);zero.setAttribute('y1',0);zero.setAttribute('y2',26);zero.setAttribute('stroke','#bbb');svg.append(zero);if(r[9]!==null&&r[10]!==null)line(r[9],r[10],'#087e8b',2);if(r[8]!==null){const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',x(r[8]));dot.setAttribute('cy',13);dot.setAttribute('r',4);dot.setAttribute('fill','#087e8b');svg.append(dot)}td.append(svg)}tr.append(td)});body.append(tr)})}render();</script></html>'''
    target = output/'evidence/all_contrasts_explorer.html'
    with target.open('x',encoding='utf-8') as handle:
        handle.write(template.replace('__DATA__',data).replace('__OPTIONS__',json.dumps(options)))
    write_json(output/'explorer_manifest.json',dict(scope='ethics_all',rows=len(frame),contrasts_per_model=52,
        path=str(target.relative_to(output)),sha256=sha256(target),data_source='statistics/paired_contrasts.csv.gz',
        data_source_sha256=sha256(output/'statistics/paired_contrasts.csv.gz'),status='candidate_pending_human_review'))
    return target
