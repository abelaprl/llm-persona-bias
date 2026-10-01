"""Question-led chart vocabulary: model-dot profiles, effects, IQR and ECDF."""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .gender_lm_reference_data import MODELS, MODEL_LABELS, COLORS, PERSONAS, SCENARIOS, persona

# Reuse the established label dictionary without importing the old render pipeline.
_tree = ast.parse(Path(__file__).with_name("restyle_gender_lm_gsm8k.py").read_text(encoding="utf-8"))
LABELS = next(ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in n.targets))
LABELS.update({"any_gender": "Ada marker gender", "explicit_user": "Rujukan pengguna eksplisit",
               "quoted_or_hypothetical": "Kutipan / hipotetis", "unknown": "Referen belum pasti",
               "scenario_overlap": "Istilah juga ada di soal"})


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 11,
                         "legend.fontsize": 10, "figure.titlesize": 17, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


def finish(fig, title, xlabel="", note="", median=False):
    fig.suptitle(title, x=.51, y=.98, fontweight="bold", color="#162032")
    if xlabel:
        fig.text(.56, .104, xlabel, ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[m], label=MODEL_LABELS[m], markersize=6) for m in MODELS]
    if median:
        handles.append(Line2D([], [], marker="D", linestyle="", color="#182233", label="Median 6 model", markersize=6))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .045), ncol=len(handles), frameon=False)
    fig.text(.5, .012, note, ha="center", fontsize=10, color="#526078")


def tidy(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")


def profile(data, marker_ids, title, scenarios=None):
    """Each panel contains all six model colors; prevalence always stays in percent."""
    style()
    scenarios = scenarios or ["N", *SCENARIOS]
    single = len(scenarios) == 1
    shape = (1, 2) if single else (2, len(scenarios))
    height = max(6.5, .54 * len(marker_ids) + 2.5) if single else max(8.2, .95 * len(marker_ids) + 3)
    fig, axes = plt.subplots(*shape, figsize=(18 if single else 22, height), squeeze=False)
    fig.subplots_adjust(left=.19 if single else .15, right=.985, top=.86, bottom=.20,
                        wspace=.58 if single else .13, hspace=.26)
    for gi, gender in enumerate(["Man", "Woman"]):
        for si, setting in enumerate(scenarios):
            ax = axes[0, gi] if single else axes[gi, si]
            block = data.loc[data.persona.eq(persona(gender, setting)) & data.marker_id.isin(marker_ids)]
            for mi, model in enumerate(MODELS):
                model_data = block.loc[block.model_id.eq(model)].set_index("marker_id").reindex(marker_ids)
                # Offset vertically only: equal values retain their exact x position.
                offsets = [-.40, -.26, -.12, .12, .26, .40]
                dots = ax.scatter(model_data.value, np.arange(len(marker_ids)) + offsets[mi],
                                  c=COLORS[model], s=18 if single else 12.25, alpha=.95,
                                  linewidths=.2, edgecolors="white", zorder=3)
                dots.set_gid("model:" + model)
            for y, marker in enumerate(marker_ids):
                values = block.loc[block.marker_id.eq(marker), "value"].dropna()
                if len(values) == 6:
                    ax.scatter(values.median(), y, marker="D", c="#182233", s=9 if single else 6.25, zorder=4)
            ax.set_title(f"{gender} · {setting}", fontweight="bold", pad=10)
            ax.set_xlim(-3, 103)
            ax.set_xticks([0,25,50,75,100] if single else [0,50,100])
            ax.set_yticks(range(len(marker_ids)), [LABELS.get(m, m) for m in marker_ids])
            if not single and si:
                ax.tick_params(labelleft=False)
            ax.set_ylim(len(marker_ids)-.4, -.6)
            tidy(ax)
    finish(fig, title, "Respons berpenjelasan dengan ≥1 hit (%)", median=True,
           note="Deskriptif · kategori dapat overlap · N yang diulang bukan observasi tambahan · marker kandidat")
    return fig


def effects(data, marker_ids, title, comparison, measurement):
    style()
    settings = ["I1", "I2", "I3"] if comparison == "implicit" else SCENARIOS
    genders = ["Woman − Man"] if comparison == "direct" else ["Man", "Woman"]
    height = max(7, .40*len(marker_ids)+2.7) if len(genders)==1 else max(10, .73*len(marker_ids)+2.7)
    fig, axes = plt.subplots(len(genders), len(settings), figsize=(22, height), squeeze=False)
    fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.84, wspace=.14, hspace=.26)
    numeric = measurement == "density"
    if numeric:
        bound = 1.
    else:
        all_values = data[["plot_value", "plot_ci_low", "plot_ci_high"]].to_numpy().ravel()
        all_values = all_values[np.isfinite(all_values)]
        bound = max(1., float(np.max(np.abs(all_values))) * 1.06) if len(all_values) else 1.
    has_na = data.plot_value.isna().any()
    for gi, gender in enumerate(genders):
        for si, setting in enumerate(settings):
            ax = axes[gi, si]
            block = data.loc[data.panel_gender.eq(gender) & data.panel_scenario.eq(setting)]
            ax.axvline(0, c="#64748b", lw=1)
            if has_na:
                ax.axvspan(1.05*bound, 1.24*bound, color="#f1f5f9", zorder=0)
                ax.text(1.145*bound, -.65, "NA", ha="center", color="#64748b", fontsize=9)
            for yi, marker in enumerate(marker_ids):
                group = block.loc[block.marker_id.eq(marker)]
                for mi, model in enumerate(MODELS):
                    row = group.loc[group.model_id.eq(model)]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    y = yi + (mi-2.5)*.065
                    value = r.plot_value
                    if pd.isna(value):
                        dots = ax.scatter(1.145*bound, y, marker="x", c=COLORS[model], s=22, linewidths=1)
                        dots.set_gid("unavailable")
                        continue
                    significant = pd.notna(r.adjusted_p) and r.adjusted_p < .05
                    alpha = .95 if significant else .40
                    if not numeric and np.isfinite([r.plot_ci_low, r.plot_ci_high]).all():
                        ax.plot([r.plot_ci_low, r.plot_ci_high], [y,y], color=COLORS[model], alpha=alpha, lw=.8)
                    changed = float(r.get("n_changed", 0))
                    size = 19 + 7*np.log10(max(0, changed)+1) if numeric else 23
                    dots = ax.scatter(value, y, c=COLORS[model], alpha=alpha, s=size,
                                      edgecolors="white", linewidths=.4, zorder=3)
                    dots.set_gid("model:"+model)
            suffix = " − N" if comparison == "neutral" else " − E" if comparison == "implicit" else ""
            ax.set_title((gender + " · " if comparison != "direct" else "") + setting + suffix, fontweight="bold")
            ax.set_xlim(-1.04*bound, (1.24 if has_na else 1.04)*bound)
            ax.set_xticks(np.linspace(-bound, bound, 5) if numeric else [-bound, 0, bound])
            if not numeric:
                ax.set_xticklabels([f"{-bound:.1f}", "0", f"{bound:.1f}"])
            ax.set_yticks(range(len(marker_ids)), [LABELS.get(m, m) for m in marker_ids])
            ax.tick_params(labelleft=si==0)
            ax.set_ylim(len(marker_ids)-.3, -.9 if has_na else -.6)
            tidy(ax)
    unit = "Matched rank-biserial" if numeric else "Delta prevalence (poin persentase) · garis = CI 95% marginal"
    direction = "Woman − Man" if comparison == "direct" else "persona − N" if comparison == "neutral" else "implicit − explicit"
    finish(fig, title, f"{unit} · {direction}",
           note="Pekat: p Holm < 0,05; pudar: lainnya · NA bukan nol · dukungan pasangan & statistik lengkap pada CSV")
    return fig


def verbosity(data):
    style()
    fig, axes = plt.subplots(1,2,figsize=(18,7))
    fig.subplots_adjust(left=.065,right=.98,top=.83,bottom=.24,wspace=.16)
    maximum = data.q75.max()*1.05
    for ax, gender in zip(axes, ["Man", "Woman"]):
        for mi, model in enumerate(MODELS):
            ordered = [persona(gender, s) for s in ["N", *SCENARIOS]]
            block = data.loc[data.model_id.eq(model)].set_index("persona").reindex(ordered)
            x = np.arange(5)+(mi-2.5)*.035
            ax.vlines(x, block.q25, block.q75, colors=COLORS[model], alpha=.7, linewidth=1.6)
            ax.plot(x, block["median"], "o-", color=COLORS[model], markersize=4, lw=1.2)
        ax.set_title(gender, fontweight="bold")
        ax.set_xticks(range(5), ["N", *SCENARIOS])
        ax.set_ylim(0, maximum)
        ax.set_ylabel("Jumlah kata: median dan IQR")
        tidy(ax)
    finish(fig,"Panjang penjelasan menurut persona dan skenario",note="Sumber frozen: verdict-only = 0 kata; missing/teknis tidak masuk · IQR bukan confidence interval")
    return fig


def coverage(status, support):
    style()
    fig, axes = plt.subplots(1,2,figsize=(18,6.5),gridspec_kw={"width_ratios":[1.5,1]})
    fig.subplots_adjust(left=.13,right=.92,top=.82,bottom=.22,wspace=.23)
    shown = status.loc[status.text_status.eq("substantive_explanation")]
    values = shown.pivot(index="model_id",columns="persona",values="percent").reindex(index=MODELS,columns=PERSONAS)
    # Absent status rows mean zero observed status count, not missing responses.
    totals = status.groupby(["model_id","persona"]).total.first().unstack().reindex(index=MODELS,columns=PERSONAS)
    values = values.fillna(0).where(totals.notna())
    paired = support.pivot(index="model_id",columns="condition",values="pair_percent").reindex(index=MODELS,columns=SCENARIOS)
    names = ["N","M·E","W·E","M·I1","W·I1","M·I2","W·I2","M·I3","W·I3"]
    for ai, (ax, table, labels, title) in enumerate(zip(axes,[values,paired],[names,SCENARIOS],
            ["Penjelasan tersedia / seluruh respons (%)", "Pasangan berpenjelasan Woman–Man (%)"])):
        img = ax.imshow(table, vmin=0,vmax=100,cmap="Blues",aspect="auto")
        for y in range(6):
            for x in range(len(labels)):
                v=table.iloc[y,x]
                ax.text(x,y,"NA" if pd.isna(v) else f"{v:.1f}",ha="center",va="center",
                        color="white" if v>65 else "#172033",fontsize=10)
        ax.set_title(title,pad=15,fontweight="bold")
        ax.set_xticks(range(len(labels)),labels)
        ax.set_yticks(range(6),[MODEL_LABELS[m] for m in MODELS] if ai==0 else [])
    fig.colorbar(img, ax=axes, fraction=.015,pad=.025,label="Persen")
    fig.suptitle("Cakupan teks dan dukungan perbandingan langsung", y=.97,fontweight="bold")
    fig.text(.53,.06,"M = Man; W = Woman · denominator 1.514 respons atau pasangan per sel · status lengkap tersedia pada CSV",ha="center",fontsize=10)
    return fig


def diversity(data):
    style()
    fig,axes=plt.subplots(2,5,figsize=(22,8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.985,top=.83,bottom=.22,wspace=.15,hspace=.30)
    for gi,gender in enumerate(["Man","Woman"]):
        for si,setting in enumerate(["N",*SCENARIOS]):
            ax=axes[gi,si]
            for model in MODELS:
                values=data.loc[data.model_id.eq(model)&data.persona.eq(persona(gender,setting)),"category_count"].dropna()
                if len(values):
                    ax.step(range(12),[100*values.le(x).mean() for x in range(12)],where="post",color=COLORS[model],lw=1.7)
            ax.set_title(f"{gender} · {setting}",fontweight="bold")
            ax.set_xlim(-.2,11.2); ax.set_ylim(-2,102); ax.set_xticks([0,2,4,6,8,10,11])
            tidy(ax)
    fig.supylabel("Respons dengan jumlah jenis ≤ x (%)", x=.025, y=.52, fontsize=12)
    finish(fig,"Keragaman kategori penanda gender dalam satu respons","Jumlah kategori berbeda (0–11)",
           note="Deskriptif; dipengaruhi panjang respons dan kamus · nol hit masuk, teks tidak tersedia tetap NA · kurva ke kanan: lebih banyak kategori")
    return fig
