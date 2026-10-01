"""Offline navigation for pre-rendered SVG views; no client-side statistics."""
import html
import json
from pathlib import Path

from .gender_lm_presentation_v2_data import GROUP_LABELS

EXPLORER_SCRIPT = r"""
const views = JSON.parse(document.getElementById('view-data').textContent);
const el = id => document.getElementById(id);
function chooseView(items, group, mode, persona, scenario) {
  return items.find(v => v.group_id === group && v.mode === mode &&
    (mode === 'profile' ? v.scenario === scenario : true) &&
    (mode === 'neutral' || mode === 'implicit' ? v.persona === persona : true));
}
function render() {
  const mode = el('mode').value;
  el('persona-label').hidden = !(mode === 'neutral' || mode === 'implicit');
  el('scenario-label').hidden = mode !== 'profile';
  const view = chooseView(views, el('group').value, mode, el('persona').value, el('scenario').value);
  if (!view) { el('caption').textContent = 'Tampilan tidak tersedia.'; el('chart').removeAttribute('src'); el('table').innerHTML = ''; return; }
  el('chart').src = view.svg_data_uri;
  el('chart').alt = view.view_id;
  el('caption').textContent = view.caption;
  el('view-title').textContent = view.view_id;
  el('table').innerHTML = view.table_html;
  el('download').href = view.svg_data_uri;
  el('download').download = view.view_id + '.svg';
}
function openHash() {
  const id = decodeURIComponent(location.hash.slice(1));
  const view = views.find(v => v.view_id === id);
  if (view) {
    el('group').value = view.group_id; el('mode').value = view.mode;
    if (view.persona) el('persona').value = view.persona;
    if (view.scenario) el('scenario').value = view.scenario;
  }
  render();
}
['group', 'mode', 'persona', 'scenario'].forEach(id => el(id).addEventListener('change', render));
window.addEventListener('hashchange', openHash);
openHash();
"""


def write_explorer(views, output_path):
    """Write only a new HTML file; table_html must be generated with escaped cells."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if not views or len({v["view_id"] for v in views}) != len(views):
        raise ValueError("Empty or duplicate views")
    if any(not v["svg_data_uri"].startswith("data:image/svg+xml;base64,") for v in views):
        raise ValueError("Only embedded SVG images are supported")
    payload = json.dumps(views, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    options = "".join(f'<option value="{group}">{group} · {html.escape(GROUP_LABELS[group])}</option>'
                      for group in dict.fromkeys(v["group_id"] for v in views))
    document = f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>Gender ETHICS — seluruh marker</title>
<style>body{{font:16px system-ui,sans-serif;margin:24px;color:#172033}}main{{max-width:1100px;margin:auto}}
nav{{display:flex;gap:16px;flex-wrap:wrap;position:sticky;top:0;background:white;padding:14px 0}}
select{{padding:8px;max-width:360px}}img{{display:block;width:16cm;max-width:100%;height:auto;margin:auto}}
figcaption{{white-space:pre-line}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border:1px solid #dce2ea}}
.scroll{{overflow:auto;max-height:65vh}}.note{{background:#f1f5f9;padding:14px}}[hidden]{{display:none!important}}
@media print{{nav,details{{display:none}}img{{width:16cm}}}}</style></head><body><main>
<h1>Penanda linguistik Gender ETHICS</h1>
<p class="note">41 marker candidate · warna = model · profil dan tiga perbandingan.
Tidak ada perhitungan ulang statistik. Jumlah pasangan dan Holm ada pada tabel.
Profil memakai seluruh penjelasan tersedia; efek memakai pasangan eligible.
CI marginal bukan CI rank-biserial atau CI terkoreksi Holm.</p>
<nav><label>Kelompok <select id="group">{options}</select></label>
<label>Tampilan <select id="mode"><option value="direct">Man − Woman</option><option value="profile">Profil absolut</option>
<option value="neutral">Persona − netral</option><option value="implicit">Implisit − eksplisit</option></select></label>
<label id="persona-label">Persona <select id="persona"><option>Man</option><option>Woman</option></select></label>
<label id="scenario-label">Skenario <select id="scenario"><option>E</option><option>I1</option><option>I2</option><option>I3</option></select></label></nav>
<h2 id="view-title"></h2><figure><img id="chart" alt="Grafik marker"><figcaption id="caption"></figcaption></figure>
<p><a id="download">Unduh SVG tampilan ini</a> · <a href="README.md">Metode, keterbatasan, dan cara ekspor PNG</a> ·
<a href="examples.html">Contoh ungkapan dan konteks</a> · <a href="visual_coverage.csv">Cakupan seluruh marker</a></p>
<details><summary>Tabel pembentuk dan konteks jumlah mentah</summary><div id="table" class="scroll"></div></details>
<script type="application/json" id="view-data">{payload}</script><script>{EXPLORER_SCRIPT}</script>
</main></body></html>"""
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(document)
