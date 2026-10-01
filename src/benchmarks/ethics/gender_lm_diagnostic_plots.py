"""Presentation-only diagnostics using frozen effect estimates and intervals."""
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .gender_lm_diagnostic_data import CONDITIONS, SCOPES, INDICATORS, PERSONAS
from .restyle_gender_lm_gsm8k import LABELS, load_reference_style

MODEL_LABELS, GENDER_COLORS = load_reference_style()
SCOPE_LABELS = dict(zip(SCOPES, ["All", "CS", "DE", "JU", "VI"]))
SCOPE_COLORS = dict(zip(SCOPES, ["#172554", "#2563EB", "#C2410C", "#047857", "#7E22CE"]))
UNITS = {"count": "Selisih rerata jumlah marker", "density": "Selisih rerata marker / 100 kata",
         "presence": "Selisih prevalensi (poin persentase)"}


def draw_estimate(ax, row, y, color):
    """Draw every finite effect, including zero and nonsignificant estimates."""
    if pd.isna(row.effect):
        ax.text(0, y, "NA", ha="center", va="center", fontsize=13)
        return "na"
    alpha = 1 if pd.notna(row.adjusted_p) and row.adjusted_p < .05 else .5
    if pd.notna(row.ci_low) and pd.notna(row.ci_high):
        ax.plot([row.ci_low, row.ci_high], [y, y], color=color, alpha=alpha, linewidth=1.5)
    ax.plot(row.effect, y, "o", markersize=7, color=color, alpha=alpha,
            markerfacecolor="none" if row.n_changed < 10 else color)
    return "numeric"


def effect_bound(data):
    values = data[["effect", "ci_low", "ci_high"]].to_numpy(float)
    return max(.001, float(np.nanmax(np.abs(values))) * 1.12) if np.isfinite(values).any() else 1


def point_audit(kinds, data):
    result = dict(numeric_points=kinds.count("numeric"), na_points=kinds.count("na"), source_rows=len(data))
    assert result["numeric_points"] + result["na_points"] == len(data)
    return result


def style_effect_axis(ax, bound):
    ax.axvline(0, color="#475569", linewidth=.8, zorder=0)
    ax.set_xlim(-bound, bound)
    ax.grid(axis="x", color="#E2E8F0", linewidth=.65)
    ax.locator_params(axis="x", nbins=3)
    ax.tick_params(axis="x", labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)


def subset_figure(data, models=MODEL_LABELS):
    marker = data.marker_id.iloc[0]
    bound = effect_bound(data)
    fig, axes = plt.subplots(4, len(models), figsize=(max(10, 4.7 * len(models)), 13.8), squeeze=False)
    kinds = []
    for column, (model, label) in enumerate(models.items()):
        for row_index, condition in enumerate(CONDITIONS):
            ax = axes[row_index, column]
            part = data.loc[data.model_id.eq(model) & data.condition.eq(condition)].set_index("scope")
            assert set(part.index) == set(SCOPES) and part.index.is_unique
            labels = []
            for y, scope in zip(range(4, -1, -1), SCOPES):
                point = part.loc[scope]
                kinds.append(draw_estimate(ax, point, y, SCOPE_COLORS[scope]))
                labels.append(f"{SCOPE_LABELS[scope]} {int(point.n_pairs)}/{int(point.n_changed)}")
            ax.set_yticks(range(4, -1, -1), labels, fontsize=12.5)
            ax.set_ylim(-.65, 4.65)
            style_effect_axis(ax, bound)
            if row_index == 0:
                ax.set_title(textwrap.fill(label, 19), fontsize=18, fontweight="bold", pad=14)
            if column == 0:
                ax.set_ylabel(condition, fontsize=18, fontweight="bold", labelpad=10)
    fig.subplots_adjust(left=.065, right=.99, top=.89, bottom=.13, hspace=.38, wspace=.85)
    fig.suptitle("Sensitivitas subset · Woman − Man\n" + LABELS[marker], fontsize=20, fontweight="bold", y=.98)
    fig.supxlabel(UNITS[data.measurement.iloc[0]] + " · CI nominal 95% · skala sama pada seluruh panel", y=.082, fontsize=17.5)
    fig.text(.02, .04, "All: gabungan; CS: commonsense; DE: deontology; JU: justice; VI: virtue. Label baris = n_pairs / n_changed.", fontsize=13.5)
    fig.text(.02, .013, "Candidate · pekat: p Holm < 0,05; pudar: lainnya · kosong: n_changed < 10 · NA ≠ 0 · subset bukan sampel independen dari overall.", fontsize=13.5)
    return fig, point_audit(kinds, data)


def normalization_figure(data, models=MODEL_LABELS):
    marker = data.marker_id.iloc[0]
    fig, axes = plt.subplots(len(models), 2, figsize=(23, max(7, 3.25 * len(models) + 2)), squeeze=False)
    kinds = []
    for column, measurement in enumerate(["count", "density"]):
        measured = data.loc[data.measurement.eq(measurement)]
        bound = effect_bound(measured)
        for row_index, (model, label) in enumerate(models.items()):
            ax = axes[row_index, column]
            part = measured.loc[measured.model_id.eq(model)].set_index("condition")
            assert set(part.index) == set(CONDITIONS) and part.index.is_unique
            labels = []
            for y, condition in zip(range(3, -1, -1), CONDITIONS):
                point = part.loc[condition]
                kinds.append(draw_estimate(ax, point, y, "#264653"))
                labels.append(f"{condition}  {int(point.n_pairs)}/{int(point.n_changed)}")
            ax.set_yticks(range(3, -1, -1), labels, fontsize=14)
            ax.set_ylim(-.55, 3.55)
            style_effect_axis(ax, bound)
            if row_index == 0:
                ax.set_title(UNITS[measurement], fontsize=18, fontweight="bold", pad=16)
            if column == 0:
                ax.set_ylabel(textwrap.fill(label, 17), fontsize=16, fontweight="bold", labelpad=15)
    fig.subplots_adjust(left=.15, right=.985, top=.92, bottom=.10, hspace=.42, wspace=.35)
    fig.suptitle("Sensitivitas normalisasi panjang · Woman − Man\n" + LABELS[marker], fontsize=20, fontweight="bold", y=.985)
    fig.text(.03, .055, "Label skenario = n_pairs / n_changed · unit count dan density berbeda; bandingkan arah dan CI, bukan besar angka lintas kolom.", fontsize=13.5)
    fig.text(.03, .032, "Candidate · CI nominal 95% · pekat: p Holm < 0,05; pudar: lainnya · kosong: n_changed < 10 · semua hasil ditampilkan.", fontsize=13.5)
    fig.text(.03, .009, "Normalisasi panjang bukan kontrol kausal; perbedaan arah tidak membuktikan bahwa efek hanya disebabkan verbosity.", fontsize=13.5)
    return fig, point_audit(kinds, data)


def heat_cells(ax, values, labels, vmax, cmap, low=None):
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
    for (y, x), value in np.ndenumerate(values):
        ax.text(x, y, labels[y][x] if np.isfinite(value) else "NA", ha="center", va="center",
                fontsize=12.5, color="white" if np.isfinite(value) and value > .55 * vmax else "#111827")
        if low is not None and low[y, x]:
            ax.add_patch(Rectangle((x-.48, y-.48), .96, .96, fill=False, edgecolor="#DC2626", linewidth=1.7))
    return image


def reference_figure(data, models=MODEL_LABELS):
    fig, axes = plt.subplots(math.ceil(len(models)/3), 3, figsize=(25, 15), squeeze=False)
    vmax = max(1, data.percent.max())
    for ax, (model, label) in zip(axes.flat, models.items()):
        part = data.loc[data.model_id.eq(model)].set_index(["persona", "indicator"])
        values = np.array([[part.loc[(p, i), "percent"] for i in INDICATORS] for p in PERSONAS])
        labels = [[f"{int(part.loc[(p, i), 'n_responses'])}\n{part.loc[(p, i), 'percent']:.2f}%" for i in INDICATORS] for p in PERSONAS]
        image = heat_cells(ax, values, labels, vmax, "Blues")
        ticks = []
        for persona in PERSONAS:
            short = "N" if persona == "neutral" else ("Man " if persona.startswith("man_") else "Woman ") + ("E" if "explicit" in persona else "I" + persona[-1])
            ticks.append(f"{short} · n={int(part.loc[(persona, 'any_gender'), 'n_available'])}")
        ax.set_yticks(range(9), ticks, fontsize=14)
        for text, persona in zip(ax.get_yticklabels(), PERSONAS):
            text.set_color(GENDER_COLORS.get(persona.split("_")[0], "#334155"))
        ax.set_xticks(range(5), ["Ada marker\ngender", "Pengguna\neksplisit", "Kutipan /\nhipotetis", "Rujukan\nunknown", "Overlap\nskenario"], fontsize=14)
        ax.set_title(label, fontsize=18, fontweight="bold", pad=12)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=.09, right=.96, top=.92, bottom=.14, hspace=.28, wspace=.4)
    cax = fig.add_axes([.975, .22, .008, .6])
    fig.colorbar(image, cax=cax, label="% penjelasan tersedia")
    fig.suptitle("Konteks rujukan gender · unit analisis satu respons", fontsize=20, fontweight="bold", y=.98)
    fig.text(.025, .083, "Isi sel = jumlah respons dan % dari n penjelasan tersedia. Satu respons dihitung sekali per kolom; kolom boleh bertumpang tindih.", fontsize=14)
    fig.text(.025, .050, "Unknown tetap unknown. Overlap istilah dengan skenario bukan bukti coreference atau adopsi persona; kategori berasal dari aturan otomatis.", fontsize=14)
    fig.text(.025, .018, "Candidate · bukan hasil anotasi manusia · respons tanpa penjelasan tidak dijadikan nol · nol hit bukan jaminan tidak ada konstruk.", fontsize=14)
    return fig, dict(source_rows=len(data))


def support_overview(data, models=MODEL_LABELS):
    fig, axes = plt.subplots(1, 2, figsize=(23, 8.5))
    for ax, column, title, cmap in zip(axes, ["pair_percent", "low_support_percent"],
                                     ["Pasangan penjelasan eligible", "Marker dengan n_changed < 10"], ["Blues", "Oranges"]):
        indexed = data.set_index(["model_id", "condition"])
        values = np.array([[indexed.loc[(m, c), column] for c in CONDITIONS] for m in models])
        labels = []
        for model in models:
            row = []
            for condition in CONDITIONS:
                point = indexed.loc[(model, condition)]
                count, total = (point.n_pairs, point.total_pairs) if column == "pair_percent" else (point.low_support_markers, point.marker_count)
                row.append(f"{int(count)}/{int(total)}\n{point[column]:.1f}%")
            labels.append(row)
        heat_cells(ax, values, labels, 100, cmap)
        ax.set_yticks(range(len(models)), list(models.values()), fontsize=15.5)
        ax.set_xticks(range(4), CONDITIONS, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(.02, .17, .99, .90), w_pad=3)
    fig.suptitle("Dukungan data · Woman − Man pada kondisi sepadan", fontsize=20, fontweight="bold", y=.98)
    fig.text(.025, .08, "Kiri: pasangan eligible / seluruh soal. Kanan: marker low-support / marker terdaftar, satu ukuran utama per marker.", fontsize=14)
    fig.text(.025, .035, "Candidate · low-support berarti sedikit pasangan berbeda, bukan bukti kesetaraan · N tidak dijumlahkan lintas marker.", fontsize=14)
    return fig, dict(source_rows=len(data))


def support_detail(data, ids, models=MODEL_LABELS):
    fig, axes = plt.subplots(math.ceil(len(models)/3), 3, figsize=(25, max(9, 1.05*len(ids)+5)), squeeze=False)
    for ax, (model, label) in zip(axes.flat, models.items()):
        indexed = data.loc[data.model_id.eq(model)].set_index(["marker_id", "condition"])
        pairs = np.array([[indexed.loc[(m, c), "n_pairs"] for c in CONDITIONS] for m in ids], float)
        changed = np.array([[indexed.loc[(m, c), "n_changed"] for c in CONDITIONS] for m in ids], float)
        values = np.divide(100*changed, pairs, out=np.full_like(changed, np.nan), where=pairs>0)
        labels = [[f"{int(changed[y,x])}\n/{int(pairs[y,x])}" for x in range(4)] for y in range(len(ids))]
        heat_cells(ax, values, labels, 100, "Blues", low=changed<10)
        ax.set_yticks(range(len(ids)), [textwrap.fill(LABELS[m], 22) for m in ids], fontsize=14)
        ax.set_xticks(range(4), CONDITIONS, fontsize=16)
        ax.set_title(label, fontsize=18, fontweight="bold")
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.14, hspace=.30, wspace=.65)
    fig.suptitle("Dukungan per marker · Woman − Man", fontsize=20, fontweight="bold", y=.975)
    fig.text(.02, .073, "Isi sel = n_changed / n_pairs; warna = % pasangan yang berbeda. Bingkai merah: n_changed < 10. NA ≠ 0.", fontsize=14)
    fig.text(.02, .032, "Candidate · satu ukuran utama per marker · jumlah pasangan berbeda bukan besar efek atau ukuran validitas detector.", fontsize=14)
    return fig, dict(source_rows=len(data))
