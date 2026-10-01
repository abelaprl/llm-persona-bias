"""Display existing gender spans and two explicitly purposive response triplets."""
import html
from pathlib import Path

import pandas as pd

from .gender_lm_reference_data import SOURCE


def inline_example(output: Path, example_id: str):
    """Show a verified triplet inside the notebook, not only behind a link."""
    responses = pd.read_csv(output / "example_responses.csv", keep_default_na=False)
    spans = pd.read_csv(output / "example_spans.csv", keep_default_na=False)
    responses = responses.loc[responses.example_id.eq(example_id)]
    spans = spans.loc[spans.example_id.eq(example_id)]
    if len(responses) != 3:
        raise ValueError("Inline evidence requires one complete triplet")
    first = responses.iloc[0]
    cards = [f'<div class="inline-example"><h3>{html.escape(example_id)} · {html.escape(first["mode"])}</h3>',
             f'<p>{html.escape(first.model_id)} · {html.escape(first.subset)} · {html.escape(first.question_id)}</p>',
             '<p>Ilustrasi purposif, bukan sampel representatif; referen gender belum divalidasi.</p>']
    for row in responses.itertuples(index=False):
        hits = spans.loc[spans.response_id.eq(row.response_id)].sort_values("start")
        if len(hits):
            hit = hits.iloc[0]
            start, end = int(hit.start), int(hit.end)
            if hit.surface_name != "explanation_text" or row.explanation_text[start:end] != hit.text:
                raise ValueError("Inline span does not match explanation")
            excerpt = (html.escape(row.explanation_text[max(0, start - 150):start]) + "<mark>"
                       + html.escape(row.explanation_text[start:end]) + "</mark>"
                       + html.escape(row.explanation_text[end:end + 150]))
            note = f"{hit.marker_id} · {hit.rule_id} · explanation_text {start}:{end}"
        else:
            excerpt = html.escape(row.explanation_text[:300])
            note = "Tidak ada span gender candidate pada penjelasan ini."
        cards.append(f'<h4>{html.escape(row.persona)}</h4><blockquote>{excerpt}</blockquote>'
                     f'<p>{html.escape(note)}<br>ID: <code>{html.escape(row.response_id)}</code></p>')
    cards.append(f'<p><a href="examples.html#{html.escape(example_id)}">Penjelasan utuh dan konteks triplet</a></p></div>')
    return "".join(cards)


def build_examples(output: Path):
    chunks = []
    for chunk in pd.read_csv(SOURCE / "marker_spans_reviewed.csv.gz", chunksize=100000):
        chunks.append(chunk.loc[chunk.marker_id.str.startswith("gender_")].copy())
    spans = pd.concat(chunks, ignore_index=True)
    forms = spans.groupby(["model_id", "persona", "marker_id", "text"], dropna=False).size().reset_index(name="span_hits")
    forms.to_csv(output / "expression_forms.csv", index=False)
    columns = ["response_id", "model_id", "persona", "subset", "question_id", "explanation_text"]
    surfaces = pd.read_csv(SOURCE / "response_surfaces.csv.gz", usecols=columns, keep_default_na=False)
    keys = ["model_id", "subset", "question_id", "persona"]
    lookup = surfaces.set_index(keys)
    if not lookup.index.is_unique:
        raise ValueError("Duplicate response surface keys")
    examples, matches = [], []
    for example_id, mode in [("contoh_1", "direct"), ("contoh_2", "implicit")]:
        candidates = spans.loc[spans.persona.str.contains("explicit" if mode == "direct" else "implicit")]
        candidates = candidates.sort_values(["model_id", "subset", "question_id", "persona", "marker_id"])
        found = False
        for hit in candidates.itertuples(index=False):
            gender = hit.persona.split("_")[0]
            personas = ["neutral", "man_explicit_1", "woman_explicit_1"] if mode == "direct" else ["neutral", gender + "_explicit_1", hit.persona]
            wanted = [(hit.model_id, hit.subset, hit.question_id, p) for p in personas]
            if not all(key in lookup.index and lookup.loc[key, "explanation_text"].strip() for key in wanted):
                continue
            for key in wanted:
                row = lookup.loc[key].to_dict()
                row.update(dict(zip(keys, key)), example_id=example_id, mode=mode)
                examples.append(row)
                evidence = spans.loc[spans.response_id.eq(row["response_id"])].copy()
                evidence["example_id"] = example_id
                for span in evidence.itertuples(index=False):
                    if span.surface_name != "explanation_text" or row["explanation_text"][int(span.start):int(span.end)] != span.text:
                        raise ValueError("Span offset does not match the source explanation")
                matches.append(evidence)
            found = True
            break
        if not found:
            raise ValueError(f"No complete purposive triplet for {mode}")
    responses = pd.DataFrame(examples)
    selected_spans = pd.concat(matches, ignore_index=True)
    responses.to_csv(output / "example_responses.csv", index=False)
    selected_spans.to_csv(output / "example_spans.csv", index=False)
    cards = []
    for example_id, rows in responses.groupby("example_id", sort=False):
        cards.append(f'<h2 id="{html.escape(example_id)}">{html.escape(example_id)}: {html.escape(rows["mode"].iloc[0])}</h2>')
        cards.append(f"<p>{html.escape(rows.model_id.iloc[0])} · {html.escape(rows.subset.iloc[0])} · {html.escape(rows.question_id.iloc[0])}</p>")
        for row in rows.itertuples(index=False):
            hits = selected_spans.loc[selected_spans.example_id.eq(example_id) & selected_spans.response_id.eq(row.response_id)]
            if len(hits):
                first = hits.sort_values("start").iloc[0]
                start, end = int(first.start), int(first.end)
                quote = (html.escape(row.explanation_text[max(0, start - 150):start]) + "<mark>"
                         + html.escape(row.explanation_text[start:end]) + "</mark>"
                         + html.escape(row.explanation_text[end:end + 150]))
                note = html.escape(first.marker_id + " · " + first.rule_id)
            else:
                quote, note = html.escape(row.explanation_text[:300]), "Tidak ada span gender candidate pada penjelasan ini."
            cards.append(f"<h3>{html.escape(row.persona)}</h3><p>Cuplikan: “{quote}”</p><p>{note}</p>"
                         f"<details><summary>Penjelasan utuh · {html.escape(row.response_id)}</summary><p class='response'>"
                         f"{html.escape(row.explanation_text)}</p></details>")
    document = """<!doctype html><html lang="id"><meta charset="utf-8"><title>Contoh ungkapan gender</title>
<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;line-height:1.6}.response{white-space:pre-wrap}mark{background:#ffedac}</style>
<h1>Contoh ungkapan dan konteks</h1><p>Dua triplet dipilih purposif: hit gender pertama dalam urutan model/subset/soal/persona/marker
yang mempunyai tiga penjelasan lengkap. Ini ilustrasi, bukan sampel representatif atau estimasi prevalensi.
Nama file reviewed tidak berarti audit manusia selesai. Hit dapat merujuk soal/kutipan, bukan pengguna.
Sorotan diperiksa terhadap offset explanation_text; kutipan hanya cuplikan, teks utuh tersedia di bawah.</p>
<p><a href="explorer.html#B08_direct_overall">Grafik B08</a> · <a href="explorer.html#B09_direct_overall">Grafik B09</a> ·
<a href="expression_forms.csv">Bentuk ungkapan seluruh enam model</a> · <a href="example_spans.csv">Provenance span</a></p>
""" + "".join(cards) + "</html>"
    (output / "examples.html").write_text(document, encoding="utf-8")
    return {"example_count": 2, "example_responses": len(responses), "verified_example_spans": len(selected_spans),
            "expression_form_rows": len(forms), "selection": "purposive_first_complete_gender_hit_triplets"}
