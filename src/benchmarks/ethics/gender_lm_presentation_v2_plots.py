"""Print-size model-dot plots; no extraction, inference, or file writes."""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .gender_lm_presentation_v2_data import PROTOTYPE_MARKERS, MARKER_GROUPS, GROUP_LABELS, view_data
from .gender_lm_reference_data import MODELS, MODEL_LABELS, COLORS, SCENARIOS, PERSONAS
from .gender_lm_reference_plots import LABELS

PRINT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def plot_direct_prototype(data: pd.DataFrame) -> Figure:
    """Render the complete B02 direct grid without pooling models or jittering x."""
    keys = ["panel_scenario", "marker_id", "model_id"]
    required = set(keys + ["display_effect", "display_ci_low", "display_ci_high",
                           "display_unit", "display_direction"])
    if not required.issubset(data.columns):
        raise ValueError("Missing display columns")
    expected = {(scenario, marker, model) for scenario in SCENARIOS
                for marker in PROTOTYPE_MARKERS for model in MODELS}
    actual = set(data[keys].itertuples(index=False, name=None))
    if actual != expected or data.duplicated(keys).any():
        raise ValueError("Missing, extra or duplicate display row")
    if not (data.display_unit.eq("occurrences_per_100_words").all()
            and data.display_direction.eq("Man - Woman").all()):
        raise ValueError("Prototype requires direct density contrasts")
    numeric = data[["display_effect", "display_ci_low", "display_ci_high"]].to_numpy(dtype=float)
    if np.isinf(numeric).any() or (data.display_ci_low > data.display_ci_high).any():
        raise ValueError("Invalid display effect or CI")
    finite = numeric[np.isfinite(numeric)]
    largest = float(np.max(np.abs(finite))) if len(finite) else 0.
    bound = largest * 1.12 if largest else .1
    labels = dict(LABELS, stance_self_correction="Koreksi diri", stance_distancing="Distancing")
    offsets = [-.30, -.18, -.06, .06, .18, .30]
    na_count = int(data.display_effect.isna().sum())
    ci_na_count = int((data.display_effect.notna()
                       & data[["display_ci_low", "display_ci_high"]].isna().any(axis=1)).sum())

    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, 22 / 2.54))
        fig.subplots_adjust(left=.235, right=.95, top=.905, bottom=.29, wspace=.20, hspace=.30)
        for panel_index, (ax, scenario_id) in enumerate(zip(axes.flat, SCENARIOS)):
            ax.set_title(scenario_id, fontsize=10, fontweight="bold", pad=9)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e2e8f0", linewidth=.6)
            ax.axvline(0, color="#64748b", linewidth=.8, zorder=1)
            for y in np.arange(.5, len(PROTOTYPE_MARKERS) - .5):
                ax.axhline(y, color="#f1f5f9", linewidth=.7, zorder=0)
            for model_index, model_id in enumerate(MODELS):
                rows = data.loc[data.model_id.eq(model_id) & data.panel_scenario.eq(scenario_id)]
                rows = rows.set_index("marker_id").loc[PROTOTYPE_MARKERS]
                y = np.arange(len(PROTOTYPE_MARKERS)) + offsets[model_index]
                available = rows.display_effect.notna().to_numpy()
                ci_available = (available & rows.display_ci_low.notna().to_numpy()
                                & rows.display_ci_high.notna().to_numpy())
                lines = ax.hlines(y[ci_available], rows.display_ci_low[ci_available],
                                  rows.display_ci_high[ci_available], color=COLORS[model_id], linewidth=.7)
                lines.set_gid("ci:" + model_id)
                dots = ax.scatter(rows.display_effect[available], y[available],
                                  color=COLORS[model_id], s=10, linewidths=0, zorder=3)
                dots.set_gid("model:" + model_id)
                for missing_y in y[~available]:
                    ax.text(1.015, missing_y, "NA", transform=ax.get_yaxis_transform(),
                            color=COLORS[model_id], fontsize=9, va="center", clip_on=False)
            ax.set_yticks(range(len(PROTOTYPE_MARKERS)),
                          [textwrap.fill(labels[m], width=15) for m in PROTOTYPE_MARKERS])
            ax.tick_params(axis="y", labelleft=(panel_index % 2 == 0), length=0, pad=8)
            ax.set_ylim(len(PROTOTYPE_MARKERS) - .5, -.5)
            ax.set_xlim(-bound, bound)
            ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(-bound, bound)
            ax.set_xticks(ticks[(ticks >= -bound) & (ticks <= bound)])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#94a3b8")

        fig.suptitle("B02 · Sikap epistemik", y=.982, fontsize=12, fontweight="bold", color="#162032")
        fig.text(.5, .950, "Gender ETHICS · Man − Woman · overall", ha="center", fontsize=9)
        fig.text(.59, .238, "Delta rerata density (kemunculan/100 kata)", ha="center", fontsize=9)
        fig.text(.59, .211, "← lebih tinggi pada Woman     lebih tinggi pada Man →", ha="center", fontsize=9)
        handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[m],
                          label=MODEL_LABELS[m], markersize=np.sqrt(10)) for m in MODELS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .131),
                   ncol=3, frameon=False, columnspacing=1., handletextpad=.4)
        caption = ("Marker candidate · titik = selisih rerata respons berpasangan.\n"
                   "Garis = CI 95% marginal; bukan CI rank atau terkoreksi Holm.\n"
                   "E = eksplisit; I1–I3 = tiga narasi implisit. Uji/coverage: CSV.\n"
                   f"Estimasi NA: {na_count}; CI NA: {ci_na_count}. Nol tetap ditampilkan; NA bukan nol.")
        fig.text(.035, .029, caption, ha="left", va="bottom", fontsize=9,
                 linespacing=1.4, color="#475569")
    return fig


def audit_geometry(fig):
    """Check physical typography and bounds for every rendered view."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            outside.append(text.get_text())
    if outside or any(t.get_fontsize() < 9 for t in texts):
        raise ValueError(f"Unreadable print geometry: {outside}")
    for ax in fig.axes:
        labels = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        if any(a.overlaps(b) for i, a in enumerate(labels) for b in labels[i + 1:]):
            raise ValueError("Overlapping marker labels")
    return dict(width_cm=round(fig.get_figwidth() * 2.54, 4), height_cm=round(fig.get_figheight() * 2.54, 4),
                min_font_pt=min(t.get_fontsize() for t in texts), text_outside=outside)


def view_caption(rows, mode):
    na = int(rows["display_value" if mode == "profile" else "display_effect"].isna().sum())
    if mode == "profile":
        return ("Marker candidate · profil seluruh penjelasan tersedia; bukan pasangan.\n"
                "N = netral; E = eksplisit; I1–I3 = tiga narasi implisit.\n"
                f"NA: {na}; simbol × di kolom NA, bukan nol. Jumlah mentah: CSV.")
    ci_na = int((rows.display_effect.notna() & rows[["display_ci_low", "display_ci_high"]].isna().any(axis=1)).sum())
    return ("Marker candidate · pasangan soal sama · CI 95% marginal.\n"
            "CI bukan CI rank/terkoreksi Holm; uji dan jumlah pasangan: CSV.\n"
            f"Estimasi NA: {na}; CI NA: {ci_na}. Simbol × di kolom NA, bukan nol.")


def plot_view(rows, group_id, mode, persona=None, scenario=None):
    """Render one full-mode view; NA uses a separate labelled margin strip."""
    markers = MARKER_GROUPS[group_id]
    panels = ["N", "Man", "Woman"] if mode == "profile" else (SCENARIOS[1:] if mode == "implicit" else SCENARIOS)
    keys = ["panel_id", "marker_id", "model_id"]
    expected = {(p, m, model) for p in panels for m in markers for model in MODELS}
    if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete view grid")
    value_col = "display_value" if mode == "profile" else "display_effect"
    columns = [value_col] if mode == "profile" else [value_col, "display_ci_low", "display_ci_high"]
    numeric = rows[columns].to_numpy(dtype=float)
    if np.isinf(numeric).any() or (mode != "profile" and (rows.display_ci_low > rows.display_ci_high).any()):
        raise ValueError("Invalid effect or interval")
    finite = numeric[np.isfinite(numeric)]
    bound = float(np.max(np.abs(finite))) * 1.12 if len(finite) and np.any(finite) else .1
    units = rows.display_unit.unique()
    if len(units) != 1:
        raise ValueError("Mixed plot units")
    density = units[0] == "occurrences_per_100_words"
    label = ("Rerata density (kemunculan/100 kata)" if density else "Penjelasan dengan ≥1 hit (%)") if mode == "profile" else (
        "Delta rerata density (kemunculan/100 kata)" if density else "Selisih prevalence (poin persentase)")
    labels = dict(LABELS, stance_self_correction="Koreksi diri")
    height = max(16, 10 + 2 * len(markers))
    with plt.rc_context({**PRINT_STYLE, "savefig.bbox": None}):
        fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, height / 2.54))
        fig.subplots_adjust(left=.27, right=.94, top=.885, bottom=.26, wspace=.25, hspace=.30)
        for pi, (ax, panel) in enumerate(zip(axes.flat, panels)):
            ax.set_gid("data-panel")
            title = panel if mode == "direct" else (f"{panel} · {scenario}" if mode == "profile" and panel != "N" else panel)
            if mode == "neutral":
                title = f"{panel} − N"
            elif mode == "implicit":
                title = f"{panel} − E"
            ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e2e8f0", linewidth=.6)
            if mode != "profile":
                ax.axvline(0, color="#64748b", linewidth=.8, zorder=1)
            for y in np.arange(.5, len(markers) - .5):
                ax.axhline(y, color="#f1f5f9", linewidth=.7, zorder=0)
            missing = False
            for mi, model in enumerate(MODELS):
                block = rows.loc[rows.panel_id.eq(panel) & rows.model_id.eq(model)].set_index("marker_id").loc[markers]
                y = np.arange(len(markers)) + [-.30, -.18, -.06, .06, .18, .30][mi]
                ok = block[value_col].notna().to_numpy()
                if mode != "profile":
                    ci_ok = ok & block[["display_ci_low", "display_ci_high"]].notna().all(axis=1).to_numpy()
                    lines = ax.hlines(y[ci_ok], block.display_ci_low[ci_ok], block.display_ci_high[ci_ok], color=COLORS[model], linewidth=.7)
                    lines.set_gid(f"ci:{panel}:{model}")
                dots = ax.scatter(block[value_col][ok], y[ok], s=10, color=COLORS[model], linewidths=0, zorder=3)
                dots.set_gid(f"model:{panel}:{model}")
                if (~ok).any():
                    missing = True
                    dots = ax.scatter(np.full((~ok).sum(), 1.035), y[~ok], transform=ax.get_yaxis_transform(),
                                      marker="x", s=10, color=COLORS[model], linewidths=.7, clip_on=False)
                    dots.set_gid(f"unavailable:{panel}:{model}")
            if missing:
                ax.text(1.035, 1.01, "NA", transform=ax.transAxes, ha="center", fontsize=9)
            ax.set_yticks(range(len(markers)), [textwrap.fill(labels[m], 18) for m in markers])
            ax.tick_params(axis="y", labelleft=(pi % 2 == 0), length=0, pad=8)
            ax.set_ylim(len(markers) - .5, -.5)
            low = 0 if mode == "profile" else -bound
            ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(low, bound)
            ax.set_xticks(ticks[(ticks >= low) & (ticks <= bound)])
            ax.set_xlim(low, bound)
            for side in ["top", "left", "right"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#94a3b8")
        if len(panels) == 3:
            axes.flat[3].set_axis_off()
        desc = {"direct": "Man − Woman", "profile": f"Profil · skenario {scenario}",
                "neutral": f"{persona} − netral", "implicit": f"{persona} · implisit − eksplisit"}[mode]
        fig.suptitle(f"{group_id} · {GROUP_LABELS[group_id]}", y=.982, fontsize=12, fontweight="bold", color="#162032")
        fig.text(.5, .948, f"Gender ETHICS · {desc} · {rows.scope.iloc[0]}", ha="center", fontsize=9)
        fig.text(.58, .212, label, ha="center", fontsize=9)
        direction = {"direct": "← Woman lebih tinggi     Man lebih tinggi →",
                     "neutral": "← netral lebih tinggi     persona lebih tinggi →",
                     "implicit": "← eksplisit lebih tinggi     implisit lebih tinggi →",
                     "profile": "N ditampilkan kembali sebagai acuan, bukan observasi baru."}[mode]
        fig.text(.53, .185, direction, ha="center", fontsize=9)
        handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[m], label=MODEL_LABELS[m], markersize=np.sqrt(10)) for m in MODELS]
        if len(panels) == 3:
            axes.flat[3].legend(handles=handles, loc="center", frameon=False, ncol=1)
        else:
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .104), ncol=3, frameon=False, columnspacing=1., handletextpad=.4)
        fig.text(.035, .015, view_caption(rows, mode), va="bottom", fontsize=9, linespacing=1.35, color="#475569")
    return fig


def render_view(group_id, mode, persona=None, scenario=None, scope="overall", *, comparisons=None, summary=None):
    rows = view_data(group_id, mode, persona, scenario, scope, comparisons=comparisons, summary=summary)
    return plot_view(rows, group_id, mode, persona, scenario), rows


def plot_coverage(table):
    with plt.rc_context({**PRINT_STYLE, "savefig.bbox": None}):
        fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 20 / 2.54))
        fig.subplots_adjust(left=.28, right=.94, top=.87, bottom=.22, hspace=.65)
        for ax, kind, settings, title in zip(axes, ["explanation", "paired"], [PERSONAS, SCENARIOS],
                                           ["Penjelasan tersedia (%)", "Pasangan langsung tersedia (%)"]):
            rows = table.loc[table.kind.eq(kind)].set_index(["model_id", "setting"])
            values = np.array([[rows.loc[(model, setting), "percent"] for setting in settings] for model in MODELS])
            im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
            ax.set_title(title, fontsize=10, pad=10)
            ax.set_yticks(range(6), [MODEL_LABELS[m] for m in MODELS])
            short = ["N", "M E", "W E", "M I1", "W I1", "M I2", "W I2", "M I3", "W I3"] if kind == "explanation" else settings
            ax.set_xticks(range(len(settings)), short, rotation=45 if kind == "explanation" else 0, ha="right" if kind == "explanation" else "center")
            for y in range(6):
                for x in range(len(settings)):
                    ax.text(x, y, f"{values[y,x]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if values[y,x] > 55 else "#172033")
        cax = fig.add_axes([.30, .13, .62, .025])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
        bar.set_label("Cakupan (%)", fontsize=9)
        fig.suptitle("G00 · Cakupan penjelasan dan pasangan", fontsize=12, fontweight="bold", y=.965)
        fig.text(.035, .025, "Seluruh slot respons disertakan; angka pada sel dibulatkan.\n"
                 "M = Man; W = Woman; N = netral. Jumlah/denominator: CSV.\n"
                 "Status penjelasan bukan status parser. Pasangan: soal sama.", fontsize=9, linespacing=1.35)
    return fig
