"""Add traceable presentation supplements without changing frozen inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "result/ETHICS/reports"
SUBSETS = ["commonsense", "deontology", "justice", "virtue"]
CONDITIONS = ["explicit", "implicit_1", "implicit_2", "implicit_3"]
CONFIG = {
    "gender": (
        "Gender",
        "notebooks/gender/linguistic marker/ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb",
        "Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_workspace_v1_5_20260909_r2",
    ),
    "religion": (
        "Religion",
        "notebooks/religion/linguistic_marker_religion_ethics_thesis_v2_0_2_candidate_20260827.ipynb",
        "Religion_Persona/linguistic_markers/analysis/religion_ethics_linguistic_markers_v2_0_2_candidate_20260827",
    ),
    "regional": (
        "Regional/Cultural",
        "notebooks/regional/linguistic_marker_regional_cultural_ethics_thesis_v2_0_1_candidate_20260828.ipynb",
        "Regional_Cultural_Persona/linguistic_markers/analysis/regional_lm_v2_0_1_complete_20260828",
    ),
}
MODELS = {
    "deepseek_deepseek-r1-distill-qwen-32b": "DeepSeek R1 32B",
    "meta-llama_llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai_ministral-8b-2512": "Ministral 8B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3 Next 80B",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen3 Next 80B",
}
SURFACE = {"word_count", "avg_sentence_length", "ttr_300"}
HASHES: dict[str, str] = {}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def track(path):
    path = Path(path)
    value = digest(path)
    if str(path) in HASHES:
        assert HASHES[str(path)] == value, f"Source changed: {path}"
    HASHES[str(path)] = value
    return path


def read(path, **kwargs):
    return pd.read_csv(track(path), **kwargs)


def normalize(data, source, comparison, gender=False, family=""):
    """Retain source row numbers and use a signed metric for direction."""
    d = data.copy()
    d["source_row"] = np.arange(len(d))
    if not gender:
        d = d.loc[d.population.eq("all_visible")].copy()
        if comparison == "within":
            d["comparison"] = np.where(d.right_condition.eq("neutral"), "vs_neutral", "vs_explicit")
        else:
            d["comparison"] = comparison
        d["scale"] = d.outcome_scale
        binary = d.scale.isin(["binary", "presence"])
        d["estimate"] = d.effect
        d["signed_effect"] = d.effect_size.where(~binary, d.effect)
        d["signed_metric"] = np.where(binary, "prevalence_delta", "rank_biserial")
        d["tier"] = d.outcome_tier
        d["n_changed"] = d.n_nonzero
        d["ci_estimand"] = np.where(binary, "prevalence_delta", "median_paired_delta")
    else:
        d["comparison"] = comparison
        binary = "presence" in family
        d["scale"] = "presence" if binary else "density"
        if not binary:
            d.loc[d.outcome_id.isin(SURFACE), "scale"] = "numeric"
        d["estimate"] = d["prevalence_delta" if binary else "median_paired_delta"]
        d["signed_effect"] = d["prevalence_delta" if binary else "rank_biserial"]
        d["signed_metric"] = "prevalence_delta" if binary else "rank_biserial"
        d["effect_size_name"] = "matched_odds_ratio" if binary else "paired_rank_biserial"
        d["effect_size"] = d["matched_odds_ratio" if binary else "rank_biserial"]
        d["tier"] = "gender_specific" if "gender" in family else "core_confirmatory"
        d["n_changed"] = d["discordant" if binary else "n_nonzero"]
        d["ci_estimand"] = "prevalence_delta" if binary else "median_paired_delta"
        d["multiplicity_family"] = "direct_4" if comparison == "direct" else "within_7"
    d["source_file"] = str(source.relative_to(ROOT)).replace("\\", "/")
    d["testable"] = d.p_holm.notna() & d.n_pairs.gt(0)
    d["decision"] = np.select(
        [~d.testable, d.p_holm.lt(.05)], ["not_testable", "significant"], default="not_significant"
    )
    d["direction"] = np.select(
        [d.signed_effect.isna(), d.signed_effect.gt(0), d.signed_effect.lt(0)],
        ["unavailable", "left_higher", "right_higher"], default="zero"
    )
    d["condition"] = d["condition"] if "condition" in d else d.left_condition
    for col in ["raw_count_sum_left", "raw_count_sum_right", "identity_contrast"]:
        if col not in d:
            d[col] = np.nan
    cols = ["comparison", "model_id", "subset", "condition", "contrast_id", "identity_contrast",
            "left_persona", "right_persona", "outcome_id", "tier", "scale", "estimate",
            "ci_estimand", "ci_low", "ci_high", "signed_metric", "signed_effect",
            "effect_size_name", "effect_size", "n_pairs", "n_changed", "p_raw", "p_holm",
            "multiplicity_family", "test_used", "decision", "direction",
            "raw_count_sum_left", "raw_count_sum_right", "source_file", "source_row"]
    return d[cols].reset_index(drop=True)


def load_results(group, bundle):
    frames = []
    if group == "gender":
        for comparison, suffix in [("direct", "direct_man_woman"), ("vs_neutral", "vs_neutral"), ("vs_explicit", "vs_explicit")]:
            for family in ["numeric", "gender_marker_presence", "gender_marker_density"]:
                population = "all_text" if family == "numeric" or comparison == "direct" else "all_visible_text"
                path = bundle / f"paired_{family}_{suffix}_{population}.csv"
                frames.append(normalize(read(path), path, comparison, True, family))
    else:
        for comparison, name in [("direct", "paired_direct_identity_all_scales.csv.gz"), ("within", "paired_within_identity_all_scales.csv.gz")]:
            path = bundle / "tables" / name
            frames.append(normalize(read(path), path, comparison))
    result = pd.concat(frames, ignore_index=True)
    key = ["comparison", "model_id", "subset", "left_persona", "right_persona", "outcome_id", "scale"]
    assert not result.duplicated(key).any(), "Duplicate presentation keys"
    assert set(result.subset) == set(SUBSETS)
    assert result.p_holm.dropna().between(0, 1).all()
    assert result.signed_effect.dropna().between(-1.00000001, 1.00000001).all()
    assert result.n_changed.le(result.n_pairs).all()
    return result


def summarize(d):
    rows = []
    keys = ["comparison", "subset", "model_id", "tier", "scale", "signed_metric"]
    for key, frame in d.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key))
        row.update(total=len(frame), tested=int(frame.decision.ne("not_testable").sum()),
                   significant=int(frame.decision.eq("significant").sum()),
                   not_significant=int(frame.decision.eq("not_significant").sum()),
                   not_testable=int(frame.decision.eq("not_testable").sum()),
                   positive=int(frame.direction.eq("left_higher").sum()),
                   negative=int(frame.direction.eq("right_higher").sum()),
                   zero=int(frame.direction.eq("zero").sum()),
                   unavailable=int(frame.direction.eq("unavailable").sum()),
                   effect_min=frame.signed_effect.min(), effect_max=frame.signed_effect.max(),
                   n_pairs_min=frame.n_pairs.min(), n_pairs_max=frame.n_pairs.max(),
                   n_changed_min=frame.n_changed.min(), n_changed_max=frame.n_changed.max())
        assert row["total"] == row["significant"] + row["not_significant"] + row["not_testable"]
        rows.append(row)
    return pd.DataFrame(rows)


def heterogeneity(d):
    rows = []
    keys = ["comparison", "model_id", "left_persona", "right_persona", "outcome_id", "tier", "scale", "signed_metric"]
    for key, frame in d.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key))
        for field, direction in [("positive_subsets", "left_higher"), ("negative_subsets", "right_higher"), ("zero_subsets", "zero"), ("unavailable_subsets", "unavailable")]:
            row[field] = ";".join(frame.loc[frame.direction.eq(direction), "subset"])
        row["opposite_directions"] = bool(row["positive_subsets"] and row["negative_subsets"])
        row["subsets_available"] = int(frame.signed_effect.notna().sum())
        row["significant_subsets"] = ";".join(frame.loc[frame.decision.eq("significant"), "subset"])
        rows.append(row)
    return pd.DataFrame(rows)


def validate_source_numbers(d):
    count = 0
    for filename, frame in d.groupby("source_file"):
        source = read(ROOT / filename).iloc[frame.source_row.to_numpy()]
        for target, original in [("p_holm", "p_holm"), ("p_raw", "p_raw"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("n_pairs", "n_pairs")]:
            np.testing.assert_allclose(frame[target], source[original], rtol=0, atol=0, equal_nan=True)
            count += len(frame)
        for target, candidates in [("estimate", ["effect", "prevalence_delta", "median_paired_delta"]),
                                    ("effect_size", ["effect_size", "matched_odds_ratio", "rank_biserial"]),
                                    ("n_changed", ["n_nonzero", "discordant"])]:
            original = next(c for c in candidates if c in source)
            np.testing.assert_allclose(frame[target], source[original], rtol=0, atol=0, equal_nan=True)
            count += len(frame)
    return count


def short_identity(value):
    return {"north_american": "Amerika Utara", "east_asian": "Asia Timur", "south_asian": "Asia Selatan",
            "sub_saharan_african": "Afrika Sub-Sahara", "man": "Man", "woman": "Woman"}.get(value, value.title())


def plot_direct(d, output, title):
    index = []
    direct = d.loc[d.comparison.eq("direct") & d.outcome_id.eq("word_count")]
    for subset in SUBSETS:
        data = direct.loc[direct.subset.eq(subset)]
        models = sorted(data.model_id.unique())
        contrasts = list(data.identity_contrast.drop_duplicates())
        fig, axes = plt.subplots(2, 2, figsize=(15, 6 if len(contrasts) == 1 else 11), layout="constrained")
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#c8c8c8")
        for ax, condition, label in zip(axes.flat, CONDITIONS, ["E", "I1", "I2", "I3"]):
            selected = data.loc[data.condition.eq(condition)]
            matrix = selected.pivot(index="identity_contrast", columns="model_id", values="signed_effect").reindex(index=contrasts, columns=models)
            pvals = selected.pivot(index="identity_contrast", columns="model_id", values="p_holm").reindex(index=contrasts, columns=models)
            im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap=cmap, aspect="auto")
            labels = [" −\n".join(short_identity(x) for x in c.split("_vs_")) for c in contrasts]
            ax.set_yticks(range(len(contrasts)), labels, fontsize=9)
            ax.set_xticks(range(len(models)), [MODELS.get(x, x) for x in models], rotation=35, ha="right", fontsize=9)
            ax.set_title(label, fontweight="bold")
            for i in range(len(contrasts)):
                for j in range(len(models)):
                    v = matrix.iloc[i, j]
                    mark = "*" if pd.notna(pvals.iloc[i, j]) and pvals.iloc[i, j] < .05 else ""
                    ax.text(j, i, "NA" if pd.isna(v) else f"{v:.2f}{mark}", ha="center", va="center", fontsize=8,
                            color="white" if pd.notna(v) and abs(v) > .6 else "black")
        fig.suptitle(f"{title} · {subset.title()}\nPerbandingan langsung jumlah kata — candidate", fontsize=16, fontweight="bold")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="Rank-biserial: identitas pertama − kedua")
        fig.supxlabel("Biru: positif · Merah: negatif · Putih: nol · Abu-abu: NA\n* p Holm < 0,05; tanpa bintang bukan bukti kesetaraan. E/I1/I2/I3 tetap terpisah.", fontsize=10)
        stem = f"direct_word_count_{subset}"
        fig.savefig(output / f"{stem}.png", dpi=140)
        fig.savefig(output / f"{stem}.svg")
        plt.close(fig)
        selected = data.copy()
        selected.to_csv(output / f"{stem}.csv", index=False)
        index.append({"figure": f"{stem}.png", "source_table": f"{stem}.csv", "subset": subset,
                      "rows": len(selected), "effect_metric": "rank_biserial", "ci_estimand": "median_paired_delta"})
    pd.DataFrame(index).to_csv(output / "figure_index.csv", index=False)


def explorer(d, output, title, *, overall=False):
    """Write a standalone local explorer; escaped JSON never becomes markup."""
    columns = ["subset", "comparison", "model_id", "condition", "left_persona", "right_persona", "outcome_id", "tier", "scale",
               "estimate", "ci_estimand", "ci_low", "ci_high", "signed_metric", "signed_effect", "effect_size_name", "effect_size",
               "n_pairs", "n_changed", "p_holm", "decision", "multiplicity_family", "raw_count_sum_left", "raw_count_sum_right", "source_file", "source_row"]
    # Preserve tiny p-values and finite doubles exactly; pandas.to_json rounds.
    values = d[columns].astype(object).where(d[columns].notna(), None)
    values = values.replace({np.inf: "Infinity", -np.inf: "-Infinity"})
    payload_object = {"columns": columns, "data": values.to_numpy().tolist()}
    payload = json.dumps(payload_object, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    assert json.loads(payload) == payload_object
    page = '''<!doctype html><html lang="id"><meta charset="utf-8"><title>Penjelajah marker</title>
<style>body{font:15px system-ui;margin:28px;color:#172b40}h1{font-size:25px}label{display:inline-block;margin:8px}select{display:block;max-width:280px;padding:7px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ddd;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#eef3f8}#scroll{overflow:auto;max-height:68vh}.note{max-width:1100px;line-height:1.6}button{padding:8px;margin:5px}.candidate{color:#8b4100}</style>
<h1>__TITLE__ — penjelajah hasil linguistic marker</h1><p class="candidate">Candidate · seluruh hasil utama tersedia, termasuk tidak signifikan dan tidak dapat diuji.</p>
<p class="note">Satu observasi adalah satu respons; uji memasangkan soal dan subset yang sama pada satu model. Analisis utama mencakup teks tersedia termasuk abstain.
<b>signed_effect</b> memakai rank-biserial untuk numerik/density dan selisih prevalensi untuk presence. Positif berarti sisi kiri lebih tinggi.
<b>CI merujuk estimate/ci_estimand, bukan rank-biserial atau odds ratio.</b> Nilai nol berbeda dari NA. Density per 100 kata; presence dalam proporsi.
Holm disalin dari keluarga uji asal; mengubah filter tidak menghitung ulang p. source_row adalah indeks data CSV berbasis nol.
Ukuran sampel dan n_changed perlu dibaca bersama efek. Hasil tidak signifikan bukan bukti kesetaraan; audit manusia belum selesai.</p>
<p><a href="all_comparisons.csv">Unduh seluruh perbandingan CSV</a> · <a href="heterogeneity.csv">Arah antar-subset CSV</a> · <a href="source_manifest.json">Provenance SHA-256</a></p>
<div id="filters"></div><p id="status"></p><button id="prev">Sebelumnya</button><button id="next">Berikutnya</button><div id="scroll"><table><thead></thead><tbody></tbody></table></div>
<script id="payload" type="application/json">__DATA__</script><script>
const payload=JSON.parse(document.getElementById('payload').textContent), cols=payload.columns, data=payload.data;
const specs=['subset','comparison','outcome_id','scale','model_id','condition','tier','left_persona','right_persona','decision'], filters={}, limit=80;let page=0, filtered=[];
for(const key of specs){const i=cols.indexOf(key),label=document.createElement('label');label.textContent=key;const s=document.createElement('select');s.add(new Option('Semua',''));[...new Set(data.map(r=>r[i]))].filter(v=>v!==null).sort().forEach(v=>s.add(new Option(v,v)));label.appendChild(s);document.getElementById('filters').appendChild(label);filters[key]=s;s.onchange=()=>{page=0;render()};}
filters.subset.value='commonsense';filters.comparison.value='direct';filters.outcome_id.value='word_count';
const tr=document.createElement('tr');cols.forEach(c=>{const th=document.createElement('th');th.textContent=c;tr.appendChild(th)});document.querySelector('thead').appendChild(tr);
function render(){filtered=data.filter(r=>specs.every(k=>!filters[k].value||String(r[cols.indexOf(k)])===filters[k].value));const body=document.querySelector('tbody');body.replaceChildren();for(const row of filtered.slice(page*limit,(page+1)*limit)){const tr=document.createElement('tr');row.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v===null?'NA':typeof v==='number'&&!Number.isInteger(v)?v.toPrecision(6):v;if(cols[i]==='signed_effect'&&v!==null){td.style.background=v>0?'#e4f0fa':v<0?'#fbe8e5':'#f5f5f5'}tr.appendChild(td)});body.appendChild(tr)}document.getElementById('status').textContent=`${filtered.length} hasil cocok dari ${data.length}; halaman ${page+1} / ${Math.max(1,Math.ceil(filtered.length/limit))}. Maksimum ${limit} baris per halaman; semua halaman tersedia.`;document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=(page+1)*limit>=filtered.length;}
document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script></html>'''
    if overall:
        page = page.replace("['subset','comparison'", "['comparison'")
        page = page.replace("filters.subset.value='commonsense';", "")
        page = page.replace(' · <a href="heterogeneity.csv">Arah antar-subset CSV</a>', '')
        page = page.replace('penjelajah hasil linguistic marker', 'penjelajah linguistic marker keseluruhan ETHICS')
        page = page.replace('Satu observasi adalah satu respons;', 'Empat subset ETHICS digabung dengan bobot sama per respons. Satu observasi adalah satu respons;')
    (output / "explorer.html").write_text(page.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload), encoding="utf-8")


def examples(manifest, d, output):
    """Select full paired examples without reading ground truth or p-values."""
    marker_dir = Path(manifest.get("source_marker_dir", manifest.get("marker_dir")))
    path = marker_dir / "response_level_markers.csv.gz"
    markers = read(path, usecols=["model_id", "persona", "question_id", "subset", "text_available", "word_count"], dtype={"question_id": str})
    direct = d.loc[d.comparison.eq("direct") & d.condition.eq("explicit")]
    model = sorted(direct.model_id.unique())[0]
    pair = direct.loc[direct.model_id.eq(model), ["left_persona", "right_persona"]].drop_duplicates().sort_values(["left_persona", "right_persona"]).iloc[0]
    selected = []
    for subset in SUBSETS:
        pool = markers.loc[markers.model_id.eq(model) & markers.subset.eq(subset) & markers.text_available.eq(True)]
        left = pool.loc[pool.persona.eq(pair.left_persona)].set_index("question_id")
        right = pool.loc[pool.persona.eq(pair.right_persona)].set_index("question_id")
        assert left.index.is_unique and right.index.is_unique
        qid = sorted(set(left.index) & set(right.index))[0]
        for persona, row in [(pair.left_persona, left.loc[qid]), (pair.right_persona, right.loc[qid])]:
            selected.append(dict(model=model, subset=subset, question_id=qid, persona=persona, word_count=float(row.word_count)))
    canonical = Path(manifest.get("source_input_csv", manifest.get("canonical_csv")))
    track(canonical)
    lookup = {(r["model"], r["subset"], r["question_id"], r["persona"]): r for r in selected}
    module_path = track(ROOT / "src/core/linguistic_markers/surface.py")
    spec = importlib.util.spec_from_file_location("presentation_surface", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    found = set()
    for chunk in pd.read_csv(canonical, usecols=["model", "subset", "question_id", "persona_folder", "response_content_full"], dtype={"question_id": str}, chunksize=4000):
        for row in chunk.loc[chunk.model.eq(model) & chunk.persona_folder.isin(pair.tolist())].itertuples(index=False):
            key = (row.model, row.subset, row.question_id, row.persona_folder)
            if key in lookup:
                assert key not in found
                surface = module.prepare_text_surface(row.response_content_full, "ethics")
                assert surface.text_available and surface.word_count == lookup[key]["word_count"], key
                lookup[key]["lexical_text"] = surface.lexical_text
                lookup[key]["source_file"] = str(canonical.relative_to(ROOT))
                found.add(key)
    assert len(found) == len(selected)
    (output / "paired_examples.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


GUIDE = """## Panduan membaca hasil dan peta pertanyaan penelitian

| Pertanyaan | Perbandingan | Kesimpulan yang didukung |
|---|---|---|
| Apakah persona mengubah bahasa terhadap netral? | E/I1/I2/I3 − N | Perbedaan pada skenario yang diuji; I−N juga memuat warmup/riwayat dialog |
| Apakah penyajian identitas berbeda? | I1/I2/I3 − E | Perbedaan antarskenario; ketiga implisit tetap terpisah |
| Apakah identitas menghasilkan bahasa berbeda? | Identitas pertama − kedua pada E/I1/I2/I3 yang sama | Uji langsung antaridentitas, bukan perbandingan dua tingkat signifikansi terhadap N |

**Cara membaca angka.** Count adalah jumlah kemunculan; density adalah frekuensi per 100 kata pada setiap respons; presence adalah proporsi respons yang memuat marker. Satu respons adalah satu observasi. Teks abstain tetap masuk; teks tidak tersedia adalah NA, bukan nol. Coverage teks berbeda dari commit rate jawaban.

Rank-biserial merangkum arah/besar perbedaan numerik berpasangan. Pada tabel tambahan, `estimate` dan CI adalah median selisih berpasangan untuk numerik/density, atau selisih prevalensi untuk presence. **CI tersebut bukan CI rank-biserial atau matched odds ratio.** Arah presence dibaca dari selisih prevalensi, bukan tanda odds ratio. `n_pairs` menunjukkan dukungan pasangan; `n_changed` menunjukkan pasangan nonzero/diskordan.

P Holm dibaca dalam keluarga uji sumber yang tercantum pada tabel. Filter tampilan tidak mengubah uji atau koreksi. Tanpa signifikansi bukan berarti setara; efek besar pada sedikit pasangan berubah perlu dibaca hati-hati. Perbedaan efek antar-model/subset pada penyajian ini bersifat deskriptif, bukan uji interaksi.

**Warna dan status.** Grafik tambahan memakai biru untuk positif, merah untuk negatif, putih untuk nol, dan abu-abu untuk NA; bintang menandai p Holm < 0,05. Grafik historis dapat memakai palet terbalik: selalu baca colorbar. Semua hasil tetap *candidate* sampai audit manusia dan kesesuaian metode dengan Bab IV dituntaskan.
"""


def supplement_cells(group, output, title):
    relative = str(output.relative_to(ROOT)).replace("\\", "/")
    cells = [nbformat.v4.new_markdown_cell("""## Hasil lengkap: arah, besar efek, dan dukungan pasangan

Bagian ini melengkapi gambar sebelumnya dengan seluruh perbandingan yang tersedia dari bundle sumber. Tabel dipisah menurut subset, model, kelompok marker, dan skala. `positive/negative/zero` menghitung **semua** arah terestimasi; `significant/not_significant/not_testable` adalah pembagian terpisah. Proporsi hasil signifikan tidak mengukur besar bias dan tidak dipakai untuk memeringkat kelompok persona.

Penjelajah lokal menyediakan filter untuk seluruh marker dan tiga perbandingan. Tabel tidak dipotong berdasarkan signifikansi. Nomor `source_row` merupakan indeks data CSV berbasis nol; sumber dan SHA-256 tersedia pada manifest.
""")]
    cells.append(nbformat.v4.new_code_cell(f'''from pathlib import Path
import json
import hashlib
import html
import os
import pandas as pd
from IPython.display import display, Markdown, HTML, Image
PRESENT_ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / 'src').is_dir() and (p / 'result').is_dir())
PRESENT = PRESENT_ROOT / {relative!r}
present_hashes = json.loads((PRESENT / 'artifact_hashes.json').read_text(encoding='utf-8'))
for present_name, present_expected in present_hashes.items():
    with (PRESENT / present_name).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == present_expected, present_name
def present_link(name, label):
    target = os.path.relpath(PRESENT / name, Path.cwd()).replace('\\\\', '/')
    return f'<a href="{{html.escape(target, quote=True)}}">{{html.escape(label)}}</a>'
def present_details(frame, label):
    display(HTML('<details><summary>' + html.escape(label) + '</summary><div style="overflow:auto;max-height:600px">' + frame.to_html(index=False, na_rep='NA', float_format=lambda x: f'{{x:.5g}}') + '</div></details>'))
display(HTML(present_link('explorer.html', 'Buka penjelajah interaktif seluruh hasil (browser lokal)') + ' · ' + present_link('all_comparisons.csv', 'CSV lengkap') + ' · ' + present_link('source_manifest.json', 'Provenance dan SHA-256')))
present_summary = pd.read_csv(PRESENT / 'summary.csv')
for present_comparison, present_label in [('vs_neutral','Terhadap netral'), ('vs_explicit','Implisit terhadap eksplisit'), ('direct','Langsung antaridentitas')]:
    present_part = present_summary.loc[present_summary.comparison.eq(present_comparison)]
    display(Markdown(f'**{{present_label}}** — {{int(present_part.total.sum()):,}} perbandingan; {{int(present_part.not_significant.sum()):,}} tidak signifikan; {{int(present_part.not_testable.sum()):,}} tidak dapat diuji. Jumlah ini mencakup kelompok marker berbeda; tidak ada p atau effect size gabungan.'))
    for present_subset in ['commonsense','deontology','justice','virtue']:
        present_details(present_part.loc[present_part.subset.eq(present_subset)], f'{{present_label}} · {{present_subset}} · seluruh model dan kelompok marker')
'''))
    cells.append(nbformat.v4.new_markdown_cell("""### Perbandingan langsung dan konteks panjang respons

Jumlah kata ditampilkan sebagai konteks verbositas, dipilih karena fungsi pengukurannya, bukan karena nilai p atau besar efek. Setiap gambar mempertahankan semua pasangan identitas dan seluruh model, dengan E/I1/I2/I3 terpisah. Angka pada sel adalah rank-biserial identitas pertama dikurangi kedua; bintang memakai p Holm asal. Tidak ada CI pada heatmap: CI median selisih jumlah kata tersedia pada tabel tepat di bawahnya. Grafik ini **tidak menggantikan hasil marker lain**, yang seluruhnya tersedia di penjelajah.
"""))
    cells.append(nbformat.v4.new_code_cell('''present_rows = pd.read_csv(PRESENT / 'all_comparisons.csv')
for present_subset in ['commonsense','deontology','justice','virtue']:
    display(Image(filename=str(PRESENT / f'direct_word_count_{present_subset}.png'), width=1200))
    present_words = pd.read_csv(PRESENT / f'direct_word_count_{present_subset}.csv')
    display(Markdown(f'**{present_subset.title()} — jumlah kata.** Dari {len(present_words)} perbandingan, {int(present_words.direction.eq("left_higher").sum())} berarah positif, {int(present_words.direction.eq("right_higher").sum())} negatif, dan {int(present_words.direction.eq("zero").sum())} nol; {int(present_words.decision.eq("significant").sum())} melewati p Holm < 0,05. Arah merujuk pasangan pada label masing-masing; jumlah ini bukan efek gabungan.'))
    present_details(present_words[['model_id','condition','left_persona','right_persona','signed_effect','estimate','ci_estimand','ci_low','ci_high','n_pairs','n_changed','p_holm','decision']], f'Tabel angka lengkap · {present_subset}')
    display(HTML(present_link(f'direct_word_count_{present_subset}.csv', 'Tabel sumber gambar') + ' · ' + present_link(f'direct_word_count_{present_subset}.svg', 'Gambar SVG')))
'''))
    cells.append(nbformat.v4.new_markdown_cell("""### Arah yang berbeda antar-subset dan hasil tidak signifikan

Tabel berikut membandingkan arah dari model, pasangan persona, marker, dan skala yang sama pada empat subset. `opposite_directions` berarti terdapat arah positif dan negatif; nol/NA dicatat terpisah. Ini pemeriksaan deskriptif, bukan pengujian bahwa efek antar-subset berbeda secara signifikan. Perbedaan kecil di sekitar nol tetap dapat berganti tanda. Karena itu, buka angka rinci dan CI sebelum menafsirkan pembalikan arah.
"""))
    cells.append(nbformat.v4.new_code_cell('''present_heterogeneity = pd.read_csv(PRESENT / 'heterogeneity.csv')
for present_comparison in ['vs_neutral','vs_explicit','direct']:
    present_h = present_heterogeneity.loc[present_heterogeneity.comparison.eq(present_comparison)]
    display(Markdown(f'**{present_comparison}:** {int(present_h.opposite_directions.sum()):,} dari {len(present_h):,} kombinasi memiliki arah berlawanan antar-subset. Dukungan subset tersedia dicatat pada setiap baris; angka ini bukan jumlah interaksi signifikan.'))
display(HTML(present_link('heterogeneity.csv','Seluruh kombinasi, termasuk yang tidak berbalik arah')))
present_details(present_heterogeneity.loc[present_heterogeneity.outcome_id.eq('word_count')], 'Konteks jumlah kata — seluruh kombinasi dan hasil nol')
'''))
    cells.append(nbformat.v4.new_markdown_cell("""### Contoh respons berpasangan: konteks, bukan bukti representatif

Untuk setiap subset, pasangan dipilih dari satu model tetap dan satu pasangan identitas eksplisit tetap, lalu ID soal pertama secara leksikografis yang memiliki teks tersedia pada kedua sisi. Model/pasangan juga dipilih dalam urutan leksikografis. Pemilihan tidak memakai GT, p-value, atau besar efek. Teks di bawah adalah penjelasan lengkap setelah normalisasi surface yang sudah digunakan pipeline; jumlah kata dicocokkan dengan artefak marker beku.

Contoh ini membantu membaca perbedaan panjang dan konteks bahasa; bukan bukti bahwa suatu marker tepat, bukan sampel representatif, dan bukan pengganti audit semantik. Penyebutan identitas dapat merujuk tokoh soal atau warmup. Tidak ada label audit manusia yang diisi oleh penambahan ini.
"""))
    cells.append(nbformat.v4.new_code_cell('''present_examples = json.loads((PRESENT / 'paired_examples.json').read_text(encoding='utf-8'))
for present_subset in ['commonsense','deontology','justice','virtue']:
    present_pair = [r for r in present_examples if r['subset'] == present_subset]
    present_heading = f"{present_subset} · {present_pair[0]['model']} · soal {present_pair[0]['question_id']}"
    present_body = ''.join('<h4>' + html.escape(r['persona']) + ' · ' + str(int(r['word_count'])) + ' kata</h4><div style="white-space:pre-wrap">' + html.escape(r['lexical_text']) + '</div>' for r in present_pair)
    display(HTML('<details><summary>' + html.escape(present_heading) + '</summary>' + present_body + '</details>'))
display(HTML(present_link('paired_examples.json','Contoh lengkap beserta kunci dan lokasi sumber')))
'''))
    cells.append(nbformat.v4.new_markdown_cell("""### Membawa hasil ke pembahasan tesis

Mulai dari pertanyaan penelitian, sebutkan model/subset/skenario dan ukuran yang berubah, lalu jelaskan besar efek serta ketidakpastiannya. Sertakan pengecualian atau hasil tidak signifikan yang relevan. Gunakan count bersama density untuk marker frekuensi, serta prevalensi dan pasangan diskordan untuk marker jarang. Grafik konteks jumlah kata dan contoh respons tidak cukup untuk menyimpulkan perubahan marker tertentu.

Simpan tabel lengkap dan atlas sebagai lampiran. Hindari peringkat berdasarkan jumlah hasil signifikan, kesimpulan kesetaraan dari p ≥ 0,05, serta klaim mekanisme reasoning atau orientasi moral internal. Sensitivitas committed-only adalah sampel terseleksi; baca retensi pasangan dan perubahan arah bersama analisis seluruh teks. Tidak ada akurasi gabungan empat subset pada tambahan ini. Audit manusia, potensi confounding warmup, ketidakcocokan prompt deontology/virtue, dan perbedaan sign test terhadap Bab IV tetap menjadi batas klaim.
"""))
    for index, cell in enumerate(cells):
        cell.id = f"presentation-20260913-{index}"
    return cells


def integrate(group, original, output, title):
    nb = nbformat.read(original, as_version=4)
    assert not any(c.id.startswith("presentation-20260913") for c in nb.cells), "Supplement already installed"
    before = copy.deepcopy(nb.cells)
    original_bytes = original.read_bytes()
    (output / "notebook_before.ipynb").write_bytes(original_bytes)
    cells = supplement_cells(group, output, title)
    fragment = nbformat.v4.new_notebook(cells=cells)
    # Execute rich outputs in IPython without creating external kernel files.
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(original.parent)
        execution_count = 0
        for cell in fragment.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            assert result.success, captured.stderr + captured.stdout
            cell.execution_count = execution_count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
    finally:
        os.chdir(previous_cwd)
    nbformat.write(fragment, output / "executed_supplement.ipynb")
    guide = nbformat.v4.new_markdown_cell(GUIDE)
    guide.id = "presentation-20260913-guide"
    nb.cells.insert(1, guide)
    # Place direct results before release limitations, preserving historical order.
    anchor = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and (
        "## CHECKLIST, TESTING" in c.source if group == "gender" else "## 7. Gerbang rilis" in c.source))
    nb.cells[anchor:anchor] = fragment.cells
    preserved = [c for c in nb.cells if not c.id.startswith("presentation-20260913")]
    assert preserved == before
    nbformat.validate(nb)
    nbformat.write(nb, output / "notebook_after.ipynb")
    assert original.read_bytes() == original_bytes, "Notebook changed during build"
    return {"old_cells_preserved": len(before), "new_cells": len(fragment.cells) + 1,
            "execution_engine": "IPython InteractiveShell; in-memory history; new cells only",
            "new_code_cells_executed": sum(c.cell_type == "code" for c in fragment.cells),
            "new_errors": sum(o.output_type == "error" for c in fragment.cells for o in c.get("outputs", [])),
            "notebook": str(original.relative_to(ROOT)), "notebook_sha256": digest(output / "notebook_after.ipynb")}


def install_prepared(output):
    """Install only reviewed candidates whose targets still match their backups."""
    report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert report["valid"]
    for group, (_, notebook, _) in CONFIG.items():
        target = ROOT / notebook
        prepared = output / group / "notebook_after.ipynb"
        assert digest(target) == digest(output / group / "notebook_before.ipynb"), target
        assert digest(prepared) == report[group]["notebook_sha256"], prepared
        nbformat.validate(nbformat.read(prepared, as_version=4))
    for group, (_, notebook, _) in CONFIG.items():
        target = ROOT / notebook
        temporary = target.with_suffix(".presentation-tmp.ipynb")
        assert not temporary.exists()
        shutil.copyfile(output / group / "notebook_after.ipynb", temporary)
        os.replace(temporary, target)
        assert digest(target) == report[group]["notebook_sha256"]
        print(f"[{group}] Installed verified notebook", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-tables-from", type=Path,
                        help="Copy hash-verified presentation artifacts into a new run before integration")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--install-prepared", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert output.is_relative_to(REPORTS.resolve()), "Output must be a new reports directory"
    if args.install_prepared:
        install_prepared(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    all_validation = {}
    prepared = []
    for group, (title, notebook, bundle_name) in CONFIG.items():
        print(f"[{group}] Reading frozen result tables", flush=True)
        destination = output / group
        destination.mkdir()
        if args.reuse_tables_from:
            previous = args.reuse_tables_from.resolve() / group
            previous_index = json.loads((previous / "artifact_hashes.json").read_text(encoding="utf-8"))
            for name, expected in previous_index.items():
                assert digest(previous / name) == expected, name
                shutil.copyfile(previous / name, destination / name)
            shutil.copyfile(previous / "artifact_hashes.json", destination / "artifact_hashes.json")
            provenance = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
            for filename, expected in provenance["source_hashes"].items():
                assert digest(filename) == expected, filename
                HASHES[filename] = expected
            prepared.append((group, notebook, destination, title, provenance["rows"],
                             provenance["numeric_values_matched"], len(provenance["source_hashes"]), provenance["examples"]//2))
            continue
        bundle = REPORTS / bundle_name
        manifest = json.loads(track(bundle / "manifest.json").read_text(encoding="utf-8"))
        source_keys_before = set(HASHES)
        d = load_results(group, bundle)
        checked = validate_source_numbers(d)
        d.to_csv(destination / "all_comparisons.csv", index=False)
        summary = summarize(d)
        summary.to_csv(destination / "summary.csv", index=False)
        h = heterogeneity(d)
        h.to_csv(destination / "heterogeneity.csv", index=False)
        print(f"[{group}] {len(d):,} rows verified; rendering direct comparisons", flush=True)
        plot_direct(d, destination, title)
        explorer(d, destination, title)
        paired = examples(manifest, d, destination)
        source_hashes = {p: value for p, value in HASHES.items() if p not in source_keys_before or p == str(bundle / "manifest.json")}
        # Explicitly include shared surface code even if read by an earlier group.
        source_hashes[str(ROOT / "src/core/linguistic_markers/surface.py")] = digest(ROOT / "src/core/linguistic_markers/surface.py")
        (destination / "source_manifest.json").write_text(json.dumps({"result_status": "candidate", "source_bundle": str(bundle),
            "source_bundle_version": manifest.get("bundle_version", manifest.get("analysis_version")),
            "rows": len(d), "numeric_values_matched": checked, "source_hashes": source_hashes,
            "implementation_sha256": digest(__file__), "inference_recomputed": False,
            "new_figures_selection": "word_count, all models/identity pairs/conditions/subsets; independent of p/effect",
            "example_selection": "lexicographic first model, explicit identity pair and available question_id per subset",
            "examples": len(paired)}, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
        (destination / "artifact_hashes.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        prepared.append((group, notebook, destination, title, len(d), checked, len(source_hashes), len(paired)//2))
    for group, notebook, destination, title, rows, checked, source_count, pairs in prepared:
        print(f"[{group}] Executing new cells and preserving existing outputs", flush=True)
        validation = integrate(group, ROOT / notebook, destination, title)
        validation.update(rows=rows, numeric_values_matched=checked, source_files=source_count, paired_examples=pairs)
        validation["integration_implementation_sha256"] = digest(__file__)
        all_validation[group] = validation
        (destination / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    for p, expected in HASHES.items():
        assert digest(p) == expected, f"Source was modified: {p}"
    all_validation["source_files_unchanged"] = len(HASHES)
    all_validation["valid"] = True
    (output / "validation.json").write_text(json.dumps(all_validation, indent=2), encoding="utf-8")
    print(json.dumps(all_validation, indent=2), flush=True)
    if not args.prepare_only:
        install_prepared(output)


if __name__ == "__main__":
    main()
