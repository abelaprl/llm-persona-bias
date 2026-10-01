"""Local, auditable candidate reports; no answer scoring or GT inputs."""
from __future__ import annotations

import hashlib
import html
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CONDITIONS = ("E", "I1", "I2", "I3")
PERSONAS = ["neutral"] + [f"{gender}_{suffix}" for suffix in
            ("explicit_1", "implicit_1", "implicit_2", "implicit_3") for gender in ("woman", "man")]
COLORS = {"neutral": "#777777", "woman": "#2166ac", "man": "#d97706"}
UNITS = {"count": "occurrence / respons tersedia", "density": "occurrence / 100 kata",
         "presence": "prevalence (%)"}
DIFF_UNITS = dict(UNITS, presence="selisih prevalence (poin persentase)")
KEY = ["model_id", "subset", "question_id"]
CSS = """body{font:16px system-ui,sans-serif;margin:24px;color:#17212b;background:#f5f7fa}
a{color:#2166ac}h1{font-size:26px}h2{font-size:20px}h3{font-size:16px}
.notice,nav{padding:12px;background:#fff;border:1px solid #ccd4dc;margin-bottom:14px}
.triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
article{padding:14px;background:white;border:1px solid #d4dce4;border-top:4px solid var(--gender)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui,sans-serif}
mark{background:#ffe59a;color:#111}small{overflow-wrap:anywhere}select,input{padding:8px;margin:6px}
section{margin-bottom:24px}.unavailable{color:#6b7280;font-style:italic}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border:1px solid #d4dce4;text-align:left}
@media(max-width:900px){.triplet{grid-template-columns:1fr}}
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _slug(value):
    readable = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))[:65]
    return readable + "_" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def _registry(registry, records):
    if isinstance(registry, dict):
        registry = registry.get("markers", list(registry.values()))
    items = [dict(item) for item in (registry or [])]
    known = {item.get("marker_id", item.get("id")) for item in items}
    for column in records.columns:
        if column.endswith("__count") and column[:-7] not in known:
            items.append({"marker_id": column[:-7], "display_family": "Unassigned"})
    for item in items:
        item["marker_id"] = item.get("marker_id", item.get("id"))
        item["display_family"] = item.get("display_family", item.get("family", "Unassigned"))
    return items


def summarize_markers(records, registry=None):
    """Descriptive overall/subset summaries; missing measurements stay NA."""
    rows = []
    markers = _registry(registry, records)
    scopes = [("overall", records)] + list(records.groupby("subset", sort=True, observed=True))
    for scope, scope_frame in scopes:
        for (model, persona), group in scope_frame.groupby(["model_id", "persona"], observed=True, sort=True):
            for marker in markers:
                marker_id = marker["marker_id"]
                for measurement, suffix in (("count", "count"), ("density", "density"), ("presence", "present")):
                    column = f"{marker_id}__{suffix}"
                    values = pd.to_numeric(group[column], errors="coerce") if column in group else pd.Series(np.nan, index=group.index)
                    available = values.notna()
                    mean = values.mean()
                    if measurement == "presence":
                        mean *= 100
                    rows.append(dict(scope=scope, model_id=model, persona=persona, marker_id=marker_id,
                                     display_family=marker["display_family"], measurement=measurement,
                                     measurement_unit=UNITS[measurement], value=mean, total=len(group),
                                     n_available=int(available.sum()), n_unavailable=int((~available).sum()),
                                     raw_occurrences=values.sum(min_count=1) if measurement == "count" else np.nan,
                                     audit_status="candidate"))
    return pd.DataFrame(rows)


def _status_summary(records):
    rows = []
    for scope, frame in [("overall", records)] + list(records.groupby("subset", observed=True, sort=True)):
        for (model, persona), group in frame.groupby(["model_id", "persona"], observed=True, sort=True):
            total = len(group)
            for status, count in group["text_status"].value_counts(dropna=False).items():
                rows.append(dict(scope=scope, model_id=model, persona=persona, text_status=status,
                                 count=int(count), total=total, percent=100 * count / total,
                                 audit_status="candidate"))
    return pd.DataFrame(rows)


def summarize_response_lengths(records):
    """Keep known empty explanations at zero and unavailable surfaces at NA."""
    rows = []
    for scope, frame in [("overall", records)] + list(records.groupby("subset", observed=True, sort=True)):
        for (model, persona), group in frame.groupby(["model_id", "persona"], observed=True, sort=True):
            counts = pd.to_numeric(group.get("word_count", pd.Series(np.nan, index=group.index)), errors="coerce")
            eligible = group.text_status.isin(["substantive_explanation", "verdict_only"])
            # An explicit verdict-only surface has a known empty explanation.
            counts = counts.where(~group.text_status.eq("verdict_only"), 0)
            available = eligible & counts.notna() & np.isfinite(counts) & counts.ge(0)
            values = counts.loc[available]
            quantiles = values.quantile([.05, .25, .5, .75, .95])
            rows.append(dict(scope=scope, model_id=model, persona=persona, total=len(group),
                             n_available=int(available.sum()), n_unavailable=int((~available).sum()),
                             n_zero=int(values.eq(0).sum()), n_verdict_only=int(group.text_status.eq("verdict_only").sum()),
                             n_substantive=int(group.text_status.eq("substantive_explanation").sum()),
                             mean=values.mean(), min=values.min(), p05=quantiles.loc[.05],
                             q25=quantiles.loc[.25], median=quantiles.loc[.5], q75=quantiles.loc[.75],
                             p95=quantiles.loc[.95], max=values.max(), measurement="word_count",
                             measurement_unit="words_in_visible_explanation", audit_status="candidate",
                             word_count_method=";".join(sorted(group.word_count_method.dropna().astype(str).unique())) if "word_count_method" in group else "provided_word_count",
                             denominator="substantive_explanation + verdict_only; unavailable surfaces excluded"))
    return pd.DataFrame(rows)


def render_highlighted_text(text, spans):
    """Render overlapping interval evidence once, escaping every text segment."""
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    valid = []
    for span in spans:
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text):
            valid.append((start, end, str(span.get("marker_id", "")), str(span.get("rule_id", ""))))
    boundaries = sorted({0, len(text)} | {p for start, end, _, _ in valid for p in (start, end)})
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [(marker, rule) for left, right, marker, rule in valid if left <= start and right >= end]
        escaped = _escape(text[start:end])
        if active:
            title = _escape("; ".join(f"{marker}: {rule}" for marker, rule in active))
            markers = _escape(" ".join(sorted({marker for marker, _ in active})))
            escaped = f'<mark title="{title}" data-marker="{markers}">{escaped}</mark>'
        result.append(escaped)
    return "".join(result)


def _html_page(title, body, script=""):
    return ('<!doctype html><html lang="id"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_escape(title)}</title><style>{CSS}</style><body><h1>{_escape(title)}</h1>'
            '<p class="notice">Candidate — audit belum selesai; target recall 99% belum terbukti. '
            'Teks lengkap; highlight adalah bukti aturan, bukan penilaian benar–salah.</p>'
            f'{body}{script}</body></html>')


def _build_span_lookup(spans):
    """Group boxed row dictionaries once, preserving pandas groupby semantics."""
    key_columns = KEY + ["persona"]
    if spans.empty or not all(column in spans for column in key_columns):
        return {}
    valid_keys = spans[key_columns].notna().all(axis=1)
    if not valid_keys.all():
        spans = spans.loc[valid_keys]
    lookup = {}
    # One conversion retains pandas nullable/native scalar boxing without
    # repeating DataFrame.to_dict setup for every response-level group.
    for row in spans.to_dict("records"):
        key = tuple(row[column] for column in key_columns)
        group = lookup.get(key)
        if group is None:
            lookup[key] = [row]
        else:
            group.append(row)
    return lookup


def write_evidence_explorer(records, spans, output_dir, page_size=30):
    """Shard by model/subset/questions; static HTML works with file:// and no fetch."""
    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    span_lookup = _build_span_lookup(spans)
    render_columns = KEY + ["persona", "text_status", "explanation_text", "structure_text", "visible_text", "raw_text"]
    records = records.loc[:, [column for column in render_columns if column in records]]
    index_rows = []
    for (model, subset), frame in records.groupby(["model_id", "subset"], sort=True, observed=True):
        groups = list(frame.groupby("question_id", sort=True, observed=True))
        for offset in range(0, len(groups), page_size):
            page_groups = groups[offset:offset + page_size]
            filename = _slug(f"{model}_{subset}_{offset // page_size + 1}") + ".html"
            content = ['<nav><a href="../evidence_explorer.html">← Indeks semua model/subset</a></nav>',
                       '<label>Soal <input id="question" placeholder="question_id"></label>',
                       '<label>Persona <select id="persona"><option value="">Semua</option>' +
                       ''.join(f'<option>{_escape(p)}</option>' for p in PERSONAS) + '</select></label>',
                       '<label>Marker <input id="marker" placeholder="marker_id"></label>',
                       '<label>Status <input id="status" placeholder="substantive_explanation"></label>']
            for question_id, question in page_groups:
                by_persona = {str(row["persona"]): row for row in question.to_dict("records")}
                content.append(f'<section data-question="{_escape(question_id)}"><h2>{_escape(subset)} / {_escape(question_id)}</h2>')
                # Neutral is stored once in the document and shared across the four conditions.
                content.append('<label>Kondisi <select class="condition">' + ''.join(f'<option>{c}</option>' for c in CONDITIONS) + '</select></label><div class="triplet">')
                for gender in ("neutral", "woman", "man"):
                    content.append(f'<div class="gender-column" data-gender="{gender}">')
                    persona_list = ["neutral"] if gender == "neutral" else [f"{gender}_{suffix}" for suffix in ("explicit_1", "implicit_1", "implicit_2", "implicit_3")]
                    for position, persona in enumerate(persona_list):
                        row = by_persona.get(persona)
                        relevant = span_lookup.get((model, subset, question_id, persona), [])
                        status = str(row.get("text_status", "unknown_surface")) if row else "unavailable"
                        marker_ids = " ".join(sorted({str(s.get("marker_id", "")) for s in relevant}))
                        condition = "all" if gender == "neutral" else CONDITIONS[position]
                        hidden = " hidden" if gender != "neutral" and position > 0 else ""
                        content.append(f'<article style="--gender:{COLORS[gender]}" data-persona="{_escape(persona)}" data-condition="{condition}" data-status="{_escape(status)}" data-markers="{_escape(marker_ids)}"{hidden}><h3>{_escape(persona)}</h3><small>Status: {_escape(status)}</small>')
                        if row is None or status in ("missing", "technical_only", "unknown_surface"):
                            content.append('<p class="unavailable">unavailable — tidak ada penjelasan eligible untuk anggota ini.</p>')
                        if row is not None:
                            for surface_name, label in (("explanation_text", "Penjelasan / highlight"),
                                                        ("structure_text", "Surface organisasi / highlight struktur"),
                                                        ("visible_text", "Visible penuh"),
                                                        ("raw_text", "Respons asli penuh (dapat memuat channel reasoning)")):
                                selected = [s for s in relevant if s.get("surface_name", "explanation_text") == surface_name]
                                rendered = render_highlighted_text(row.get(surface_name, ""), selected)
                                if surface_name == "explanation_text":
                                    content.append(f'<p>{label}</p><pre>{rendered}</pre>')
                                else:
                                    content.append(f'<details><summary>{label}</summary><pre>{rendered}</pre></details>')
                            if relevant:
                                content.append('<details><summary>Provenance span</summary><ul>')
                                for span in relevant:
                                    fields = [f'{key}: {span.get(key, "")}' for key in ("marker_id", "rule_id", "start", "end", "surface_name", "rule_status", "flags")]
                                    content.append('<li><small>' + _escape(" | ".join(fields)) + '</small></li>')
                                content.append('</ul></details>')
                        content.append('</article>')
                    content.append('</div>')
                content.append('</div></section>')
            script = """<script>
function apply(){const q=document.querySelector('#question').value.toLowerCase(),p=document.querySelector('#persona').value,m=document.querySelector('#marker').value.toLowerCase(),s=document.querySelector('#status').value.toLowerCase();
document.querySelectorAll('section').forEach(sec=>{sec.hidden=!sec.dataset.question.toLowerCase().includes(q);const c=sec.querySelector('.condition').value;
sec.querySelectorAll('article').forEach(a=>{a.hidden=a.dataset.condition!=='all'&&a.dataset.condition!==c;
a.style.opacity=(!p||a.dataset.persona===p)&&(!m||a.dataset.markers.toLowerCase().includes(m))&&(!s||a.dataset.status.toLowerCase().includes(s))?'1':'.25';});});}
document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',apply));
</script>"""
            (evidence_dir / filename).write_text(_html_page(f"{model} — {subset}", "".join(content), script), encoding="utf-8")
            index_rows.append(dict(model_id=model, subset=subset, first_question=str(page_groups[0][0]),
                                   last_question=str(page_groups[-1][0]), questions=" ".join(str(q[0]) for q in page_groups),
                                   n_questions=len(page_groups), n_responses=sum(len(q[1]) for q in page_groups),
                                   path="evidence/" + filename))
    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / "evidence_index.csv", index=False)
    body = '<p>Filter mencari semua ID soal pada indeks; teks respons dimuat hanya saat halaman dipilih. Setiap halaman memuat paling banyak ' + str(page_size) + ' soal.</p><input id="search" placeholder="Model, subset, atau question_id"><table><thead><tr><th>Model</th><th>Subset</th><th>Soal</th><th>Respons</th><th>Halaman</th></tr></thead><tbody>'
    for row in index_rows:
        search = _escape(f'{row["model_id"]} {row["subset"]} {row["questions"]}')
        body += f'<tr data-search="{search}"><td>{_escape(row["model_id"])}</td><td>{_escape(row["subset"])}</td><td>{row["n_questions"]}</td><td>{row["n_responses"]}</td><td><a href="{row["path"]}">{_escape(row["first_question"])} … {_escape(row["last_question"])}</a></td></tr>'
    body += '</tbody></table>'
    script = "<script>document.querySelector('#search').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.toLowerCase().includes(q));});</script>"
    (output_dir / "evidence_explorer.html").write_text(_html_page("Penjelajah bukti semua respons", body, script), encoding="utf-8")
    return index


class _Figures:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.folder = self.root / "figures"
        self.folder.mkdir(exist_ok=False)
        self.rows = []

    def save(self, fig, data, name, title, scope, kind, measurement="", estimand="", denominator=""):
        stem = _slug(name)
        csv_path = self.folder / (stem + ".csv")
        data.to_csv(csv_path, index=False)
        fig.suptitle(textwrap.fill(title, width=105), fontsize=14, fontweight="bold", y=.99)
        note = "NA ≠ 0; marker hollow = low support (n_changed < 10); bentuk kondisi tetap. CI nominal 95%; target 99% belum terbukti."
        if kind == "response_length":
            note = "Kotak = IQR; garis hitam = median; titik = rerata; whisker = persentil 5–95, bukan CI. NA ≠ 0."
        elif kind == "neutral_triplet_map":
            note = "Warna = marker; bentuk = kondisi; hollow hanya jika flag low support tersedia. CSV memuat koordinat dan n_triplets."
        fig.text(.02, .014, f"CANDIDATE | {scope} | {denominator}\n{note}", fontsize=9)
        bottom_margin = .23 if kind == "neutral_triplet_map" else .17 if kind.startswith("forest_") else .12
        fig.subplots_adjust(top=.9, bottom=bottom_margin, hspace=.48, wspace=.42)
        png, svg = self.folder / (stem + ".png"), self.folder / (stem + ".svg")
        fig.savefig(png, dpi=160, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)
        self.rows.append(dict(figure_id=stem, figure_kind=kind, title=title, scope=scope,
                              measurement=measurement, estimand=estimand, denominator=denominator,
                              audit_status="candidate", png_path=png.relative_to(self.root).as_posix(),
                              svg_path=svg.relative_to(self.root).as_posix(), source_csv=csv_path.relative_to(self.root).as_posix(),
                              source_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                              png_sha256=hashlib.sha256(png.read_bytes()).hexdigest()))


def _panels(models, height=10):
    columns = min(3, max(1, len(models)))
    rows = math.ceil(max(1, len(models)) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, height * rows / 2 + 1.5), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def _marker_pages(markers):
    families = {}
    for marker in markers:
        families.setdefault(marker["display_family"], []).append(marker["marker_id"])
    packed_names, packed_ids, page = [], [], 1
    for family, identifiers in families.items():
        for start in range(0, len(identifiers), 8):
            chunk = identifiers[start:start + 8]
            if packed_ids and len(packed_ids) + len(chunk) > 8:
                yield " + ".join(packed_names), page, packed_ids
                packed_names, packed_ids, page = [], [], page + 1
            packed_names.append(family)
            packed_ids.extend(chunk)
    if packed_ids:
        yield " + ".join(packed_names), page, packed_ids


def _short_persona(persona):
    if persona == "neutral":
        return "N"
    gender = "W" if str(persona).startswith("woman") else "M"
    condition = "E" if "explicit" in str(persona) else "I" + str(persona).split("_")[-1]
    return f"{gender} {condition}"


def _condition_label(condition):
    return {"explicit_1": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}.get(str(condition), str(condition))


def _coverage_figure(book, statuses, models):
    colors = {"substantive_explanation": "#4d9221", "verdict_only": "#b8cbd9", "missing": "#555555", "technical_only": "#e6ab02", "unknown_surface": "#a676ad"}
    data = statuses.loc[statuses.scope == "overall"]
    fig, axes = _panels(models)
    all_statuses = list(colors) + [s for s in data.text_status.unique() if s not in colors]
    for ax, model in zip(axes, models):
        part = data.loc[data.model_id == model]
        bottom = np.zeros(len(PERSONAS))
        for status in all_statuses:
            values = part.loc[part.text_status == status].set_index("persona").percent.reindex(PERSONAS).fillna(0)
            ax.barh(range(len(PERSONAS)), values, left=bottom, color=colors.get(status, "#999999"), label=status)
            bottom += values.to_numpy()
        totals = part.groupby("persona", observed=True).total.first().reindex(PERSONAS).fillna(0)
        ax.set_yticks(range(len(PERSONAS)), [f"{_short_persona(p)}  N={int(totals[p])}" for p in PERSONAS])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% seluruh respons sumber")
        ax.set_title(str(model), fontsize=10)
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .07), ncol=min(5, len(labels)), fontsize=8)
    book.save(fig, data, "overall_status", "Coverage/status: semua model × sembilan persona", "overall", "coverage", estimand="source_response_status", denominator="Seluruh respons; neutral dihitung satu kali")


def _response_length_figure(book, lengths, models):
    data = lengths.loc[lengths.scope.eq("overall")]
    fig, axes = _panels(models)
    for ax, model in zip(axes, models):
        part = data.loc[data.model_id.eq(model)].set_index("persona")
        labels = []
        for position, persona in enumerate(PERSONAS):
            row = part.loc[persona] if persona in part.index else None
            available = int(row.n_available) if row is not None else 0
            total = int(row.total) if row is not None else 0
            labels.append(f"{_short_persona(persona)}  n={available}/{total}")
            if row is None or not available:
                ax.text(.02, position, "NA", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
                continue
            color = COLORS[persona.split("_")[0]]
            ax.plot([row.p05, row.p95], [position, position], color=color, linewidth=1.5)
            ax.fill_betweenx([position - .24, position + .24], row.q25, row.q75, color=color, alpha=.35)
            ax.plot([row["median"], row["median"]], [position - .24, position + .24], color="#111", linewidth=1.5)
            ax.plot(row["mean"], position, marker="o", markersize=4, color=color)
        ax.set_yticks(range(len(PERSONAS)), labels, fontsize=8)
        ax.set_ylim(len(PERSONAS) - .5, -.5)
        ax.set_xlim(left=0)
        ax.set_title(str(model), fontsize=10)
        ax.set_xlabel("Jumlah kata penjelasan visible", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    book.save(fig, data, "overall_response_lengths", "Kontrol verbosity: distribusi panjang penjelasan visible", "overall",
              "response_length", "word_count", "observed_explanation_length_distribution",
              "n tersedia / total per persona; verdict-only = 0 kata; missing/technical/unknown = NA")


def _atlas(book, summary, markers, models, *, render_subset_figures=False):
    for scope in summary.scope.unique():
        if scope != "overall" and not render_subset_figures:
            continue
        for family, page, ids in _marker_pages(markers):
            for measurement in UNITS:
                data = summary.loc[(summary.scope == scope) & (summary.measurement == measurement) & summary.marker_id.isin(ids)]
                fig, axes = _panels(models)
                finite = data.value.dropna()
                high = max(float(finite.max()), .001) if len(finite) else 1
                for panel_number, (ax, model) in enumerate(zip(axes, models)):
                    part = data.loc[data.model_id == model]
                    matrix = part.pivot(index="marker_id", columns="persona", values="value").reindex(index=ids, columns=PERSONAS)
                    cmap = plt.get_cmap("YlGnBu").copy()
                    cmap.set_bad("#e5e7eb")
                    mesh = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=high, aspect="auto")
                    for y, marker in enumerate(ids):
                        for x, persona in enumerate(PERSONAS):
                            value = matrix.loc[marker, persona]
                            label = "NA" if pd.isna(value) else f"{value:.1f}"
                            ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white" if pd.notna(value) and value > .65 * high else "#111")
                    # Shared row order makes repeated long labels unnecessary.
                    ax.set_yticks(range(len(ids)), ids if panel_number % min(3, len(models)) == 0 else [""] * len(ids), fontsize=8)
                    ax.set_xticks(range(len(PERSONAS)), [_short_persona(p) for p in PERSONAS], rotation=45, ha="right", fontsize=8)
                    for tick, persona in zip(ax.get_xticklabels(), PERSONAS):
                        tick.set_color(COLORS[persona.split("_")[0]])
                    ax.set_title(str(model), fontsize=10)
                    fig.colorbar(mesh, ax=ax, fraction=.035, pad=.025)
                title = f"Profil marker · {family} {page} · {UNITS[measurement]}"
                book.save(fig, data, f"atlas_{scope}_{family}_{page}_{measurement}", title, scope, "marker_atlas", measurement,
                          "mean_available_response" if measurement != "presence" else "response_prevalence",
                          "Per model/persona; n_available dan total tersedia di CSV; NA dipertahankan")


def _normalize_comparisons(comparisons):
    result = comparisons.copy()
    for target, alternatives in {"effect": ["mean_difference"], "ci_high": ["ci_hi"], "matched_rank_biserial": ["rank_biserial"]}.items():
        if target not in result:
            for source in alternatives:
                if source in result:
                    result[target] = result[source]
                    break
    if "n_changed" in result:
        result["low_support"] = result.n_changed < 10
    return result


def _support_marker(symbol, color, low_support=False):
    """Encode support in fill only; keep the condition/gender symbol unchanged."""
    hollow = bool(low_support) if pd.notna(low_support) else False
    return dict(marker=symbol, markerfacecolor="none" if hollow else color, markeredgecolor=color)


def _forest(book, comparisons, markers, models):
    if comparisons.empty:
        return
    data_all = _normalize_comparisons(comparisons)
    data_all = data_all.loc[data_all.scope == "overall"].copy()
    if data_all.empty:
        return
    data_all["view"] = "implicit"
    data_all.loc[data_all.right.eq("neutral"), "view"] = "neutral"
    direct = data_all.left.str.startswith("woman", na=False) & data_all.right.str.startswith("man", na=False)
    data_all.loc[direct, "view"] = "woman_man"
    data_all.loc[data_all.role.eq("interaction"), "view"] = "interaction"
    for view, view_data in data_all.groupby("view", sort=False):
        for family, page, ids in _marker_pages(markers):
            for measurement, unit in DIFF_UNITS.items():
                data = view_data.loc[view_data.marker_id.isin(ids) & view_data.measurement.eq(measurement)]
                if data.empty:
                    continue
                contrasts = list(data.contrast_id.drop_duplicates())
                # Secondary plots use six contrasts per page so offsets remain readable.
                for start in range(0, len(contrasts), 8):
                    selected = contrasts[start:start + 8]
                    source = data.loc[data.contrast_id.isin(selected)].copy()
                    fig, axes = _panels(models, height=12)
                    offsets = np.linspace(-.34, .34, len(selected)) if len(selected) > 1 else [0]
                    for panel_number, (ax, model) in enumerate(zip(axes, models)):
                        part = source.loc[source.model_id == model]
                        for contrast, offset in zip(selected, offsets):
                            group = part.loc[part.contrast_id == contrast].set_index("marker_id")
                            example = source.loc[source.contrast_id == contrast].iloc[0]
                            color = COLORS["man"] if str(example.left).startswith("man") else COLORS["woman"]
                            symbol = ["o", "s", "^", "v", "P", "X", "<", ">"][selected.index(contrast)]
                            for y, marker_id in enumerate(ids):
                                if marker_id not in group.index:
                                    continue
                                row = group.loc[marker_id]
                                value = row.get("effect", np.nan)
                                if pd.isna(value):
                                    ax.text(.01, y + offset, "NA", transform=ax.get_yaxis_transform(), fontsize=6)
                                    continue
                                lo, hi = row.get("ci_low", np.nan), row.get("ci_high", np.nan)
                                if pd.notna(lo) and pd.notna(hi):
                                    ax.plot([lo, hi], [y + offset, y + offset], color=color, linewidth=.8, alpha=.7)
                                ax.plot(value, y + offset, markersize=5, color=color,
                                        **_support_marker(symbol, color, row.get("low_support", False)))
                        ax.axvline(0, color="#555", linewidth=.8, linestyle="--")
                        ax.set_yticks(range(len(ids)), ids if panel_number % min(3, len(models)) == 0 else [""] * len(ids), fontsize=8)
                        ax.set_ylim(len(ids) - .5, -.5)
                        ax.set_title(str(model), fontsize=10)
                        ax.set_xlabel(unit, fontsize=8)
                        ax.grid(axis="y", alpha=.15)
                    labels = []
                    handles = []
                    for ix, contrast in enumerate(selected):
                        example = source.loc[source.contrast_id == contrast].iloc[0]
                        label = str(contrast).replace("__minus__", " − ").replace("woman_", "W_").replace("man_", "M_").replace("explicit_1", "E").replace("implicit_", "I")
                        labels.append(label)
                        handles.append(Line2D([], [], marker=["o", "s", "^", "v", "P", "X", "<", ">"][ix], linestyle="", color=COLORS["man"] if str(example.left).startswith("man") else COLORS["woman"]))
                    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=min(4, len(labels)), fontsize=7)
                    title = {"neutral": "Woman/Man − Neutral", "woman_man": "Woman − Man: positif = Woman lebih tinggi", "implicit": "Kontras kondisi explicit/implicit", "interaction": "Interaksi: perubahan selisih Woman − Man"}[view]
                    book.save(fig, source, f"forest_{view}_{family}_{page}_{measurement}_{start}", f"{title} · {family} {page}", "overall", "forest_" + view,
                              measurement, ";".join(source.estimand.dropna().unique()) if "estimand" in source else "mean_paired_difference",
                              "n_pairs/n_changed/CI/sample_scope per titik di source CSV; model tidak dipool")


def _triplet_maps(book, triplets, markers, models):
    if triplets is None or triplets.empty:
        return
    overall = triplets.loc[triplets.scope == "overall"]
    if not overall.sample_scope.eq("common_triplet").all():
        raise ValueError("Relative-neutral maps require sample_scope=common_triplet")
    for family, page, ids in _marker_pages(markers):
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C564B", "#000000"]
        marker_colors = dict(zip(ids, palette))
        condition_symbols = {"E": "o", "I1": "s", "I2": "^", "I3": "v"}
        for measurement in UNITS:
            data = overall.loc[overall.marker_id.isin(ids) & overall.measurement.eq(measurement)].copy()
            if data.empty:
                continue
            data["marker_color"] = data.marker_id.map(marker_colors)
            data["condition_symbol"] = data.condition.map(_condition_label).map(condition_symbols)
            fig, axes = _panels(models, height=12)
            for ax, model in zip(axes, models):
                part = data.loc[data.model_id == model]
                x = pd.to_numeric(part.d_man_neutral, errors="coerce")
                y = pd.to_numeric(part.d_woman_neutral, errors="coerce")
                limit = max(1e-6, float(pd.concat([x.abs(), y.abs()]).max())) * 1.2
                if not np.isfinite(limit):
                    limit = 1
                ax.plot([-limit, limit], [-limit, limit], color="#aaa", linewidth=.8)
                ax.axvline(0, color="#777", linewidth=.7)
                ax.axhline(0, color="#777", linewidth=.7)
                for (_, row), xv, yv in zip(part.iterrows(), x, y):
                    if pd.isna(xv) or pd.isna(yv):
                        continue
                    condition = _condition_label(row.get("condition", row.get("left_condition", "E")))
                    symbol = condition_symbols.get(condition, "o")
                    color = marker_colors[row.marker_id]
                    ax.plot(xv, yv, linestyle="", markersize=6, color=color, alpha=.85,
                            **_support_marker(symbol, color, row.get("low_support", False)))
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_xlabel("Man − Neutral · " + DIFF_UNITS[measurement], fontsize=8, color=COLORS["man"])
                ax.set_ylabel("Woman − Neutral", fontsize=8, color=COLORS["woman"])
                ax.set_title(str(model), fontsize=10)
            visible_ids = [marker for marker in ids if marker in set(data.marker_id)]
            marker_handles = [Line2D([], [], marker="o", linestyle="", color=marker_colors[marker], markersize=6) for marker in visible_ids]
            fig.legend(marker_handles, visible_ids, title="Marker (warna)", loc="lower center",
                       bbox_to_anchor=(.5, .065), ncol=2, fontsize=8, title_fontsize=9)
            condition_handles = [Line2D([], [], marker=symbol, linestyle="", color="#333", markersize=6) for symbol in condition_symbols.values()]
            fig.legend(condition_handles, list(condition_symbols), title="Kondisi (bentuk)", loc="lower center",
                       bbox_to_anchor=(.5, .145), ncol=4, fontsize=8, title_fontsize=9)
            book.save(fig, data, f"triplet_{family}_{page}_{measurement}", f"Pergeseran relatif neutral · {family} {page}", "overall", "neutral_triplet_map", measurement,
                      "additive_common_triplet_differences", "Common triplet utuh; n_triplets di CSV; dekat diagonal bukan bukti ekuivalensi")


def _condition_profiles(book, comparisons, markers, models):
    if comparisons.empty:
        return
    source = _normalize_comparisons(comparisons)
    source = source.loc[source.scope.eq("overall") & source.right.eq("neutral")].copy()
    if source.empty:
        return
    source["gender"] = source.left.str.split("_").str[0]
    source["condition"] = source.left.str.split("_", n=1).str[1].map(_condition_label)
    for family, page, ids in _marker_pages(markers):
        for measurement in UNITS:
            data = source.loc[source.marker_id.isin(ids) & source.measurement.eq(measurement)]
            if data.empty:
                continue
            for model_start in range(0, len(models), 3):
                model_group = models[model_start:model_start + 3]
                source_page = data.loc[data.model_id.isin(model_group)]
                if source_page.empty:
                    continue
                fig, axes = plt.subplots(len(ids), len(model_group), squeeze=False,
                                         figsize=(6 * len(model_group), max(6, 2.4 * len(ids) + 2)), sharey="row")
                for row_index, marker_id in enumerate(ids):
                    for column, model in enumerate(model_group):
                        ax = axes[row_index, column]
                        part = source_page.loc[source_page.model_id.eq(model) & source_page.marker_id.eq(marker_id)]
                        for gender, symbol, offset in (("woman", "o", -.07), ("man", "s", .07)):
                            selected = part.loc[part.gender.eq(gender)].set_index("condition").reindex(CONDITIONS)
                            x = np.arange(4) + offset
                            ax.plot([], [], linestyle="", marker=symbol, markersize=5, color=COLORS[gender], label=gender)
                            for position, (_, point) in enumerate(selected.iterrows()):
                                value = point.get("effect", np.nan)
                                lo, hi = point.get("ci_low", np.nan), point.get("ci_high", np.nan)
                                if pd.isna(value):
                                    continue
                                if pd.notna(lo) and pd.notna(hi):
                                    ax.plot([x[position], x[position]], [lo, hi], linewidth=1, color=COLORS[gender])
                                ax.plot(x[position], value, linestyle="", markersize=5, color=COLORS[gender],
                                        **_support_marker(symbol, COLORS[gender], point.get("low_support", False)))
                        ax.axhline(0, color="#777", linestyle="--", linewidth=.7)
                        ax.set_xticks(range(4), CONDITIONS)
                        ax.set_xlim(-.5, 3.5)
                        ax.set_title(f"{model}\n{marker_id}", fontsize=9)
                        ax.tick_params(labelsize=8)
                        ax.grid(axis="y", alpha=.15)
                        if column == 0:
                            ax.set_ylabel(DIFF_UNITS[measurement], fontsize=8)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=2)
                book.save(fig, source_page, f"conditions_{family}_{page}_{measurement}_{model_start}",
                          f"E / I1 / I2 / I3: pergeseran terhadap neutral · {family} {page}", "overall", "condition_profile",
                          measurement, "mean_paired_difference", "Kategori kondisi, bukan skala/tren dosis; CI dan n_pairs per titik di CSV")


def _audit_figure(book, audit, markers):
    data = audit.copy()
    if data.empty:
        data = pd.DataFrame([dict(marker_id=m["marker_id"], audit_status="candidate", precision=np.nan, recall=np.nan) for m in markers])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    precision = pd.to_numeric(data["precision"], errors="coerce") if "precision" in data else pd.Series(dtype=float)
    recall = pd.to_numeric(data["recall"], errors="coerce") if "recall" in data else pd.Series(dtype=float)
    measured_p, measured_r = int(precision.notna().sum()), int(recall.notna().sum())
    text = (f"{len(markers)} marker terdaftar\n\n"
            f"Baris audit dengan estimasi precision: {measured_p}\n"
            f"Baris audit dengan estimasi recall: {measured_r}\n\n"
            "Target recall ≥99% per marker belum dinyatakan terbukti oleh laporan otomatis ini.\n"
            "Lihat source CSV untuk estimasi/CI dan keputusan audit yang tersedia.\n"
            "Anotasi kosong tidak menjadi nol kesalahan; candidate tidak dipromosikan oleh grafik.")
    ax.text(.02, .95, text, transform=ax.transAxes, va="top", fontsize=13, linespacing=1.6)
    book.save(fig, data, "audit_dashboard", "Dashboard kesiapan audit", "overall", "audit", estimand="audit_evidence_availability", denominator="Unit audit mengikuti tabel; bukan seluruh respons atau jumlah span secara otomatis")


def _registered_measurements(data, markers):
    if data is None or data.empty:
        return data
    selected = {marker["marker_id"]: "presence" if marker.get("measurement") in ("binary", "presence") else marker.get("measurement", "density") for marker in markers}
    return data.loc[data.measurement.eq(data.marker_id.map(selected))].copy()


def build_report(records: pd.DataFrame, spans: pd.DataFrame, comparisons: pd.DataFrame,
                 audit: pd.DataFrame, output_dir: Path, *, registry=None, triplets=None,
                 interactions=None, coverage=None, render_subset_figures=False,
                 full_detail=False) -> pd.DataFrame:
    """Write new report assets and return their source-linked figure index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    owned = ["figures", "evidence", "evidence_explorer.html", "evidence_index.csv", "figure_index.csv",
             "report.md", "report.html", "marker_summary.csv", "report_status_coverage.csv", "response_length_summary.csv"]
    for name in owned:
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    required = KEY + ["persona", "text_status"]
    missing = [column for column in required if column not in records]
    if missing:
        raise ValueError(f"Missing report columns: {missing}")
    if records.duplicated(KEY + ["persona"]).any():
        raise ValueError("Duplicate response keys cannot be rendered as independent observations")
    if records.empty:
        raise ValueError("Cannot report an empty source")
    markers = _registry(registry, records)
    summary = summarize_markers(records, markers)
    summary.to_csv(output_dir / "marker_summary.csv", index=False)
    statuses = _status_summary(records)
    statuses.to_csv(output_dir / "report_status_coverage.csv", index=False)
    lengths = summarize_response_lengths(records)
    lengths.to_csv(output_dir / "response_length_summary.csv", index=False)
    book = _Figures(output_dir)
    models = sorted(records.model_id.unique())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    _coverage_figure(book, statuses, models)
    _response_length_figure(book, lengths, models)
    _atlas(book, summary, markers, models, render_subset_figures=render_subset_figures)
    combined = comparisons if interactions is None or interactions.empty else pd.concat([comparisons, interactions], ignore_index=True)
    display_comparisons = combined if full_detail else _registered_measurements(combined, markers)
    display_triplets = triplets if full_detail else _registered_measurements(triplets, markers)
    _forest(book, display_comparisons, markers, models)
    _condition_profiles(book, display_comparisons, markers, models)
    _triplet_maps(book, display_triplets, markers, models)
    _audit_figure(book, audit, markers)
    write_evidence_explorer(records, spans, output_dir)
    index = pd.DataFrame(book.rows)
    index.to_csv(output_dir / "figure_index.csv", index=False)
    status_counts = records.text_status.value_counts()
    report = (f"# Laporan linguistic marker ETHICS — candidate\n\n"
              f"Sumber memuat **{len(records):,} respons**, {records.model_id.nunique()} model, "
              f"{records.persona.nunique()} persona, {records.subset.nunique()} subset, dan {len(markers)} marker. "
              "Default visual adalah overall seluruh subset; setiap model tetap terpisah.\n\n"
              "Target recall ≥99% per marker belum terbukti. Semua grafik berstatus candidate; "
              "p-value dan CI tidak memvalidasi detektor. Audit manusia dan validasi parsing tetap menjadi gate analisis final.\n\n"
              "## Denominator dan satuan\n\n"
              "Seluruh respons sumber masuk coverage, termasuk missing/verdict-only. "
              "Profil marker memakai n_available setiap measurement; NA tidak diubah menjadi nol. "
              "Count adalah occurrence per respons tersedia, density per 100 kata, prevalence dalam persen; "
              "selisih presence dalam poin persentase. Raw occurrence dan denominator tersedia di marker_summary.csv.\n\n"
              "Kontrol verbosity tersedia di response_length_summary.csv dan grafik panjang penjelasan visible. "
              "Verdict-only memiliki panjang penjelasan nol kata; missing/technical-only/unknown_surface tetap NA. "
              "Grafik memakai median, IQR dan persentil 5–95; whisker bukan CI dan bukan seluruh rentang. "
              "Rerata, minimum, maksimum, n_available, n_unavailable, dan n_zero disertakan pada source CSV.\n\n"
              "| Status teks | Respons |\n|---|---:|\n" + ''.join(f"| {status} | {count} |\n" for status, count in status_counts.items()) +
              "\n## Interpretasi perbandingan\n\n"
              "Arah direct contrast selalu Woman−Man. Positif berarti nilai marker lebih tinggi pada woman. "
              "Common-triplet map menggunakan triplet utuh; neutral tidak dihitung sebagai empat observasi independen. "
              "CI adalah CI nominal 95%, bukan CI simultan Holm. n_changed <10 ditandai marker hollow (low_support), "
              "tanpa menghapus titik atau mengganti bentuk kondisi. "
              "Ukuran efek rank dan selisih rerata mempunyai estimand berbeda dan tidak saling dikurangkan.\n\n"
              "Perbedaan persona implisit adalah efek kondisi prompt beserta narasinya; bukan isolasi gender murni. "
              "Kedekatan dengan diagonal atau p>0,05 bukan bukti ekuivalensi. Tidak ada filter significant-only atau committed-only.\n\n"
              "## Akses bukti\n\n"
              "Buka [laporan visual](report.html) atau [penjelajah respons penuh](evidence_explorer.html). "
              "figure_index.csv merekam source CSV, satuan, denominator, scope, status, dan SHA256 setiap figur. "
              "marker_summary.csv tetap memuat seluruh subset dan ketiga satuan. "
              "Default render adalah overall; render_subset_figures=True menambah atlas subset. "
              "Default forest/profile/triplet memakai measurement registry (binary → presence); "
              "full_detail=True merender ketiga measurement. Family registry tetap utuh saat keluarga kecil "
              "berbagi halaman dengan maksimum delapan marker.\n\n")
    if comparisons.empty:
        report += "Perbandingan berpasangan belum tersedia pada input ini; forest tidak dibuat.\n\n"
    if triplets is None or triplets.empty:
        report += "Input common-triplet belum tersedia; peta relatif neutral tidak dibuat.\n\n"
    report += "## Indeks figur\n\n| Jenis | Scope | Satuan | PNG | Data |\n|---|---|---|---|---|\n"
    for row in index.itertuples():
        report += f"| {row.figure_kind} | {row.scope} | {row.measurement} | [PNG]({row.png_path}) | [CSV]({row.source_csv}) |\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    body = (f'<p>{len(records):,} respons · {len(models)} model · {records.persona.nunique()} persona · {len(markers)} marker</p>'
            '<nav><a href="evidence_explorer.html">Penjelajah teks lengkap</a> · <a href="figure_index.csv">Indeks + hash</a> · <a href="marker_summary.csv">Denominator marker</a></nav>'
            '<label>Scope <select id="scope">' + ''.join(f'<option>{_escape(s)}</option>' for s in index.scope.drop_duplicates()) + '</select></label>'
            '<label>Jenis <select id="kind"><option value="">Semua</option>' + ''.join(f'<option>{_escape(s)}</option>' for s in index.figure_kind.drop_duplicates()) + '</select></label>'
            '<label>Satuan <select id="unit"><option value="">Semua</option><option>count</option><option>density</option><option>presence</option><option>word_count</option></select></label>')
    for row in index.itertuples():
        body += (f'<section class="figure" data-scope="{_escape(row.scope)}" data-kind="{_escape(row.figure_kind)}" data-unit="{_escape(row.measurement)}"' + (' hidden' if row.scope != 'overall' else '') + '>'
                 f'<h2>{_escape(row.title)}</h2><p>{_escape(row.denominator)}</p><a href="{row.png_path}"><img loading="lazy" src="{row.png_path}" alt="{_escape(row.title)}" style="max-width:100%;height:auto"></a>'
                 f'<p><a href="{row.svg_path}">SVG</a> · <a href="{row.source_csv}">Source CSV (N, CI, effect)</a></p></section>')
    script = """<script>function filter(){const s=document.querySelector('#scope').value,k=document.querySelector('#kind').value,u=document.querySelector('#unit').value;document.querySelectorAll('.figure').forEach(e=>e.hidden=e.dataset.scope!==s||(k&&e.dataset.kind!==k)||(u&&e.dataset.unit!==u));}document.querySelectorAll('select').forEach(e=>e.addEventListener('change',filter));filter();</script>"""
    (output_dir / "report.html").write_text(_html_page("Atlas linguistic marker ETHICS", body, script), encoding="utf-8")
    return index
