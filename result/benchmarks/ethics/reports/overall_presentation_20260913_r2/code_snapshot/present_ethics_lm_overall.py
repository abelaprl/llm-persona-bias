"""Present and audit response-weighted ETHICS marker results in three notebooks."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.MAIN_PARSING.ETHICS.run_gender_ethics_lm_workspace import enable_marker_imports, sha256
enable_marker_imports()
from src.MAIN_PARSING.ETHICS import complete_lm_notebook_presentation as prior
from src.MAIN_PARSING.ETHICS.build_ethics_lm_overall_statistics import SOURCES, engine_for
from src.MAIN_PARSING.ETHICS.build_gender_ethics_direct_thesis_visuals_v1_2 import GENDER_MARKER_LABELS
from src.MAIN_PARSING.ETHICS.build_religion_ethics_linguistic_markers_v2 import OUTCOME_LABELS as RELIGION_LABELS
from src.MAIN_PARSING.ETHICS.build_regional_cultural_ethics_linguistic_markers_v2 import OUTCOME_LABELS as REGIONAL_LABELS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nbformat
import numpy as np
import pandas as pd

REPORTS = ROOT / "result/ETHICS/reports"
GENDER_POOL = REPORTS / "Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_20260909_r4"
GENDER_MARKERS = REPORTS / "Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0/response_level_markers.csv.gz"
CONDITIONS = ["explicit", "implicit_1", "implicit_2", "implicit_3"]
MODEL_ORDER = list(engine_for("religion").MODELS)
MODEL_LABELS = engine_for("religion").MODEL_LABELS
CORE_IDS = list(engine_for("religion").SURFACE_OUTCOMES) + list(engine_for("religion").CORE_MARKERS)
IDENTITIES = {"gender": ["man", "woman"], "religion": list(engine_for("religion").IDENTITIES), "regional": list(engine_for("regional").IDENTITIES)}
IDENTITY_LABELS = {"man": "Laki-laki", "woman": "Perempuan", **engine_for("religion").IDENTITY_LABELS, **engine_for("regional").IDENTITY_LABELS}
SHORT_IDENTITIES = {"man": "Man", "woman": "Woman", "islam": "Islam", "christian": "Kristen", "jewish": "Yahudi", "atheist": "Ateis",
                    "north_american": "Am.Utara", "east_asian": "Asia T.", "south_asian": "Asia S.", "sub_saharan_african": "Afrika SS"}
LABELS = {**RELIGION_LABELS, **REGIONAL_LABELS, **GENDER_MARKER_LABELS}
LABELS.update({"organization_bullet": "Daftar berpoin", "organization_section_header": "Subjudul", "stance_self_correction": "Koreksi diri",
               "self_reference_inclusive_we": "We inklusif", "stance_distancing": "Pengambilan jarak", "stance_hedge": "Hedging", "stance_booster": "Kepastian"})


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def verify_index(directory):
    index = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in index.items():
        assert sha256(directory / name) == expected, name
    return index


def persona_info(group, persona):
    if persona == "neutral":
        return "neutral", "neutral"
    identity = next(i for i in IDENTITIES[group] if persona.startswith(i + "_"))
    rest = persona[len(identity)+1:]
    return identity, "explicit" if rest.startswith("explicit") else rest


def persona_order(group):
    return ["neutral"] + [f"{i}_{c}" + ("_1" if group == "gender" and c == "explicit" else "") for i in IDENTITIES[group] for c in CONDITIONS]


def persona_label(group, persona):
    identity, condition = persona_info(group, persona)
    if persona == "neutral":
        return "N"
    scenario = {"explicit": "E", "implicit_1": "I1", "implicit_2": "I2", "implicit_3": "I3"}[condition]
    return f"{SHORT_IDENTITIES[identity]} {scenario}"


def load_data(group, statistics):
    pool = GENDER_POOL if group == "gender" else statistics / group
    source_index = verify_index(pool)
    source_manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    results = []
    if group == "gender":
        for population, suffix in [("all_visible", "all_text"), ("both_committed", "committed")]:
            for comparison in ["vs_neutral", "vs_explicit", "direct"]:
                for family in ["numeric", "gender_presence", "gender_density"]:
                    source = pool / f"{family}_{comparison}_{suffix}.csv"
                    data = prior.normalize(pd.read_csv(source), source, comparison, True, family)
                    data["population"] = population
                    results.append(data)
        frame = pd.read_csv(GENDER_MARKERS, low_memory=False)
        frame["text_available"] = engine_for("religion").bool_series(frame.text_available)
        frame["validated_committed"] = frame.answer_status.eq("committed")
        frame["identity"] = [persona_info(group, p)[0] for p in frame.persona]
        frame["condition"] = [persona_info(group, p)[1] for p in frame.persona]
        specific_ids = list(GENDER_MARKER_LABELS)
        openings = []
        for comparison in ["within", "direct"]:
            opening = pd.read_csv(pool / f"opening_{comparison}_all_text.csv")
            opening["population"] = "all_visible"
            openings.append(opening)
        opening = pd.concat(openings, ignore_index=True)
    else:
        frame = pd.read_csv(pool / "pooled_response_level.csv.gz", low_memory=False)
        dictionary = pd.read_csv(pool / "outcome_dictionary.csv")
        specific_ids = dictionary.loc[dictionary.outcome_tier.eq(f"{('regional_cultural' if group == 'regional' else 'religion')}_specific"), "outcome_id"].tolist()
        for comparison, name in [("within", "within.csv"), ("direct", "direct.csv")]:
            source = pool / name
            data = pd.read_csv(source)
            # normalize() retains the all_visible rows and source row indices.
            for population in data.population.unique():
                selected = data.copy()
                selected.loc[selected.population.ne(population), "population"] = "exclude"
                selected.loc[data.population.eq(population), "population"] = "all_visible"
                standardized = prior.normalize(selected, source, comparison)
                standardized["population"] = population
                results.append(standardized)
        opening = pd.read_csv(pool / "opening.csv")
    d = pd.concat(results, ignore_index=True)
    d["scale"] = d.scale.replace({"binary": "presence", "density_per_100_words": "density"})
    d["identity"] = [persona_info(group, p)[0] for p in d.left_persona]
    d["pair_label"] = [" − ".join(IDENTITY_LABELS[i] for i in c.split("_vs_")) if isinstance(c,str) and "_vs_" in c else "" for c in d.identity_contrast]
    assert set(d.subset) == {"ethics_all"}
    assert not d.duplicated(["population","model_id","left_persona","right_persona","outcome_id","scale"]).any()
    assert d.p_holm.dropna().between(0,1).all()
    return pool, d, frame, specific_ids, opening, source_index, source_manifest


def apply_style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.titlesize":13, "axes.titleweight":"bold",
                         "axes.labelsize":11, "xtick.labelsize":10, "ytick.labelsize":10, "figure.titlesize":17,
                         "figure.titleweight":"bold", "savefig.facecolor":"white", "svg.fonttype":"none"})


class FigureWriter:
    def __init__(self, output):
        self.output = output
        self.figures = output / "figures"
        self.tables = output / "figure_tables"
        self.figures.mkdir(); self.tables.mkdir()
        self.index = []

    def save(self, fig, name, table, section, caption, metric, identity="", comparison=""):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        clipped = []
        from matplotlib.text import Text
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > bounds.width+2 or box.y1 > bounds.height+2:
                clipped.append(text.get_text())
        assert not clipped, (name, clipped)
        assert not (self.figures / f"{name}.png").exists()
        table.to_csv(self.tables / f"{name}.csv", index=False)
        fig.savefig(self.figures / f"{name}.png", dpi=130)
        fig.savefig(self.figures / f"{name}.svg")
        self.index.append({"figure":f"figures/{name}.png", "svg":f"figures/{name}.svg", "table":f"figure_tables/{name}.csv",
                           "section":section, "comparison":comparison,"identity":identity,"rows":len(table),"metric":metric,
                           "caption":caption,"clipped_text":0,"aggregation":"ethics_all", "audit_status":"automated_pass_visual_pending"})
        plt.close(fig)


def effect_maps(d, writer, group):
    core = d.loc[d.population.eq("all_visible") & d.tier.eq("core_confirmatory")]
    for comparison in ["vs_neutral","vs_explicit","direct"]:
        sub = core.loc[core.comparison.eq(comparison)]
        key = "identity_contrast" if comparison == "direct" else "identity"
        for identity, data in sub.groupby(key, sort=False):
            conditions = CONDITIONS if comparison != "vs_explicit" else CONDITIONS[1:]
            fig, axes = plt.subplots(1,len(conditions), figsize=(17 if len(conditions)==4 else 15,10), sharey=True)
            cmap=plt.get_cmap("RdBu").copy();cmap.set_bad("#cccccc")
            for ax, condition in zip(axes,conditions):
                panel = data.loc[data.condition.eq(condition)]
                values = panel.pivot(index="outcome_id",columns="model_id",values="signed_effect").reindex(index=CORE_IDS,columns=MODEL_ORDER)
                p = panel.pivot(index="outcome_id",columns="model_id",values="p_holm").reindex(index=CORE_IDS,columns=MODEL_ORDER)
                assert len(panel)==len(CORE_IDS)*len(MODEL_ORDER)
                im=ax.imshow(values.to_numpy(),vmin=-1,vmax=1,cmap=cmap,aspect="auto")
                label={"explicit":"E","implicit_1":"I1","implicit_2":"I2","implicit_3":"I3"}[condition]
                ax.set_title(label + (" − N" if comparison=="vs_neutral" else " − E" if comparison=="vs_explicit" else ""))
                ax.set_xticks(range(6),[MODEL_LABELS[m] for m in MODEL_ORDER],rotation=60,ha="right")
                for i in range(len(CORE_IDS)):
                    for j in range(6):
                        if pd.notna(p.iloc[i,j]) and p.iloc[i,j]<.05:
                            ax.text(j,i,"*",ha="center",va="center",fontsize=12,color="white" if abs(values.iloc[i,j])>.65 else "black")
                np.testing.assert_allclose(np.asarray(im.get_array()),values.to_numpy(),equal_nan=True)
            axes[0].set_yticks(range(len(CORE_IDS)),[LABELS[x] for x in CORE_IDS])
            identity_label=" − ".join(IDENTITY_LABELS[i] for i in identity.split("_vs_")) if comparison=="direct" else IDENTITY_LABELS[identity]
            caption=f"{identity_label}: " + ("identitas pertama dikurangi kedua pada skenario yang sama." if comparison=="direct" else "skenario pertama dikurangi pembanding pada judul panel.")
            caption += " Warna rank-biserial; bintang p Holm < 0,05. Semua model dan hasil tidak signifikan tetap ditampilkan. CI median selisih tersedia di tabel, bukan CI rank-biserial."
            fig.suptitle(f"ETHICS · {identity_label}\n" + {"vs_neutral":"Terhadap netral","vs_explicit":"Implisit dan eksplisit","direct":"Perbandingan langsung antaridentitas"}[comparison])
            fig.subplots_adjust(left=.19,right=.9,bottom=.23,top=.87,wspace=.1)
            color_ax=fig.add_axes([.925,.3,.014,.5]);fig.colorbar(im,cax=color_ax,label="Rank-biserial")
            fig.text(.19,.055,"Biru: positif · Merah: negatif · Putih: nol · Abu-abu: NA · * p Holm < 0,05\nEmpat subset digabung pada tingkat respons; model dan E/I1/I2/I3 tetap terpisah. Hasil candidate.",fontsize=10)
            writer.save(fig,f"core_{comparison}_{identity}",data,"core",caption,"rank_biserial",identity,comparison)


def coverage_plot(frame, writer, group):
    rows=frame.groupby(["model_id","persona"]).agg(total=("question_id","size"),available=("text_available","sum"),
                committed=("validated_committed","sum"),abstain=("answer_status",lambda s:int(s.eq("abstain").sum())),missing=("answer_status",lambda s:int(s.eq("missing").sum()))).reset_index()
    # Keep canonical statuses separate from validated_committed used by sensitivity.
    rows.rename(columns={"committed":"validated_committed"},inplace=True)
    rows["committed"]=rows.total-rows.abstain-rows.missing
    for col in ["available","committed","abstain","missing"]:
        rows[f"{col}_rate"]=rows[col]/rows.total
    rows["commit_rate"]=rows.committed/rows.total
    matrix=rows.pivot(index="model_id",columns="persona",values="available_rate").reindex(index=MODEL_ORDER,columns=persona_order(group))
    fig,ax=plt.subplots(figsize=(16,5.8))
    im=ax.imshow(matrix.to_numpy()*100,cmap="Blues",vmin=0,vmax=100,aspect="auto")
    ax.set_yticks(range(6),[MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xticks(range(len(matrix.columns)),[persona_label(group,p) for p in matrix.columns],rotation=55,ha="right")
    for i in range(6):
        for j in range(len(matrix.columns)):
            ax.text(j,i,f"{100*matrix.iloc[i,j]:.1f}",ha="center",va="center",fontsize=9,color="white" if matrix.iloc[i,j]>.6 else "black")
    ax.set_title("ETHICS · Ketersediaan teks penjelasan (%)")
    fig.colorbar(im,ax=ax,label="Teks tersedia (%)",pad=.02)
    fig.subplots_adjust(left=.14,bottom=.3,right=.96,top=.9)
    writer.save(fig,"coverage",rows,"coverage","Persentase teks tersedia dari seluruh respons per model/persona. Coverage teks berbeda dari commit rate jawaban; abstain dengan teks tetap masuk analisis.","percent_available")
    return rows


def verbosity_plot(frame, writer, group):
    selected=frame.loc[frame.text_available]
    rows=selected.groupby(["model_id","persona","identity","condition"],dropna=False).agg(n=("word_count","count"),median=("word_count","median"),q1=("word_count",lambda s:s.quantile(.25)),q3=("word_count",lambda s:s.quantile(.75))).reset_index()
    fig,axes=plt.subplots(3,2,figsize=(15,12),sharex=True)
    colors=["#2864a5","#b84c35","#44815e","#8662a6"]
    for ax,model in zip(axes.flat,MODEL_ORDER):
        data=rows.loc[rows.model_id.eq(model)]
        for identity,color in zip(IDENTITIES[group],colors):
            personas=["neutral"]+[p for p in persona_order(group) if p.startswith(identity+"_")]
            series=data.set_index("persona").reindex(personas)
            ax.plot(range(5),series["median"],marker="o",color=color,label=IDENTITY_LABELS[identity])
        ax.set_title(MODEL_LABELS[model]);ax.set_ylabel("Median jumlah kata");ax.set_xticks(range(5),["N","E","I1","I2","I3"]);ax.grid(axis="y",alpha=.2)
        ax.set_ylim(bottom=0)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=len(IDENTITIES[group]),bbox_to_anchor=(.5,.025))
    fig.suptitle("ETHICS · Panjang respons sebagai konteks density")
    fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.92,hspace=.3,wspace=.22)
    writer.save(fig,"verbosity",rows,"verbosity","Median jumlah kata pada semua teks tersedia, termasuk abstain. N adalah baseline bersama yang sama pada setiap garis identitas. Sumbu per model dapat berbeda; IQR dan dukungan respons tersedia di tabel.","median_word_count")
    return rows


def specific_prevalence(frame, ids, writer, group):
    selected=frame.loc[frame.text_available]
    rows=[]
    for marker in ids:
        for (model,persona),data in selected.groupby(["model_id","persona"],sort=False):
            rows.append({"model_id":model,"persona":persona,"outcome_id":marker,"n_text":len(data),
                "raw_count":data[f"{marker}__count"].sum(),"mean_density":data[f"{marker}__density"].mean(),
                "prevalence":data[f"{marker}__present"].mean()})
    table=pd.DataFrame(rows)
    for model in MODEL_ORDER:
        data=table.loc[table.model_id.eq(model)]
        matrix=data.pivot(index="outcome_id",columns="persona",values="prevalence").reindex(index=ids,columns=persona_order(group))
        fig,ax=plt.subplots(figsize=(15, max(7, len(ids)*.32+2.5)))
        im=ax.imshow(matrix.to_numpy()*100,vmin=0,vmax=100,cmap="Blues",aspect="auto")
        ax.set_yticks(range(len(ids)),[textwrap.fill(LABELS.get(i,i.replace('_',' ')),32) for i in ids])
        ax.set_xticks(range(len(matrix.columns)),[persona_label(group,p) for p in matrix.columns],rotation=55,ha="right")
        ax.set_title(f"ETHICS · {MODEL_LABELS[model]}\nPrevalensi marker khusus identitas (%)")
        for i in range(len(ids)):
            for j in range(len(matrix.columns)):
                val=100*matrix.iloc[i,j]
                ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=9,color="white" if val>60 else "black")
        fig.colorbar(im,ax=ax,label="Respons memuat marker (%)",pad=.02)
        fig.subplots_adjust(left=.29,bottom=.22,right=.95,top=.91)
        writer.save(fig,f"specific_{MODEL_ORDER.index(model)+1:02d}",data,"specific","Prevalensi marker khusus pada seluruh teks tersedia, termasuk abstain, per model/persona. Angka deskriptif ini tidak menguji perbedaan; uji langsung, count, density, CI dan p tersedia di penjelajah/tabel. Skala warna 0–100% tetap sama.","prevalence_percent",model)
    return table


def specific_direct_maps(d, ids, writer):
    selected=d.loc[d.population.eq("all_visible") & d.comparison.eq("direct") & d.outcome_id.isin(ids)]
    for identity,data in selected.groupby("identity_contrast",sort=False):
        metrics=[m for m in ["rank_biserial","prevalence_delta"] if m in set(data.signed_metric)]
        metric_ids={m:[i for i in ids if i in set(data.loc[data.signed_metric.eq(m),"outcome_id"])] for m in metrics}
        heights=[max(4,len(metric_ids[m])) for m in metrics]
        figure_height=sum(heights)*.36+6
        fig,axes=plt.subplots(len(metrics),4,figsize=(17,figure_height),squeeze=False,gridspec_kw={"height_ratios":heights})
        cmap=plt.get_cmap("RdBu").copy();cmap.set_bad("#cccccc")
        for row,metric in enumerate(metrics):
            scale=100 if metric=="prevalence_delta" else 1
            for col,condition in enumerate(CONDITIONS):
                ax=axes[row,col]
                panel=data.loc[data.condition.eq(condition)&data.signed_metric.eq(metric)]
                values=panel.pivot(index="outcome_id",columns="model_id",values="signed_effect").reindex(index=metric_ids[metric],columns=MODEL_ORDER)
                p=panel.pivot(index="outcome_id",columns="model_id",values="p_holm").reindex(index=metric_ids[metric],columns=MODEL_ORDER)
                im=ax.imshow(scale*values.to_numpy(),vmin=-scale,vmax=scale,cmap=cmap,aspect="auto")
                label={"explicit":"E","implicit_1":"I1","implicit_2":"I2","implicit_3":"I3"}[condition]
                ax.set_title(label+" · "+("Density" if metric=="rank_biserial" else "Presence"))
                ax.set_yticks(range(len(metric_ids[metric])),[textwrap.fill(LABELS.get(i,i),32) for i in metric_ids[metric]] if col==0 else [""]*len(metric_ids[metric]))
                ax.set_xticks(range(6),[MODEL_LABELS[m] for m in MODEL_ORDER],rotation=60,ha="right")
                for i in range(len(metric_ids[metric])):
                    for j in range(6):
                        if pd.notna(p.iloc[i,j]) and p.iloc[i,j]<.05:
                            ax.text(j,i,"*",ha="center",va="center",fontsize=12,color="white" if abs(values.iloc[i,j])>.65 else "black")
                np.testing.assert_allclose(np.asarray(im.get_array()),scale*values.to_numpy(),equal_nan=True)
            fig.colorbar(im,ax=axes[row,:].tolist(),fraction=.015,pad=.015,label="Rank-biserial" if metric=="rank_biserial" else "Selisih prevalensi (pp)")
        label=" − ".join(IDENTITY_LABELS[i] for i in identity.split("_vs_"))
        fig.suptitle(f"ETHICS · {label}\nMarker khusus: perbandingan langsung")
        # Position axes explicitly before adding a separate colorbar for each scale.
        fig.subplots_adjust(left=.24,right=.88,bottom=2.6/figure_height,top=.88,wspace=.1,hspace=.65)
        # Colorbar axes created before subplots_adjust need relocation beside their row.
        for row,metric in enumerate(metrics):
            bounds=axes[row,-1].get_position()
            fig.axes[len(metrics)*4+row].set_position([.91,bounds.y0,.013,bounds.height])
        footer=fig.text(.24,.3/figure_height,"Biru: identitas pertama lebih tinggi · Merah: identitas kedua lebih tinggi · * p Holm < 0,05\nDensity memakai rank-biserial; presence memakai selisih prevalensi (poin persentase). Skala berbeda ditampilkan terpisah.\nSeluruh ETHICS; hasil candidate. Count, dukungan pasangan, CI dan nilai p tersedia pada tabel sumber.",fontsize=10)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in axes.flat:
            for label in ax.get_xticklabels():
                assert not footer.get_window_extent(renderer).overlaps(label.get_window_extent(renderer)), identity
        writer.save(fig,f"specific_direct_{identity}",data,"specific_direct","Identitas pertama dikurangi kedua pada E/I1/I2/I3. Panel density: rank-biserial; panel presence: selisih prevalensi dalam poin persentase. Colorbar berbeda, bukan skala effect size yang sama. CI numerik adalah median selisih; CI presence adalah selisih prevalensi. Bintang p Holm asal; hasil tidak signifikan tidak dihapus.","separate_rank_biserial_and_prevalence_pp",identity,"direct")


def opening_plot(frame, writer, group):
    available=frame.loc[frame.text_available]
    rows=available.groupby(["model_id","persona","opening_category"],dropna=False).size().rename("count").reset_index()
    rows["proportion"]=rows["count"]/rows.groupby(["model_id","persona"])["count"].transform("sum")
    categories=sorted(rows.opening_category.dropna().unique())
    fig,axes=plt.subplots(3,2,figsize=(17, max(13,len(persona_order(group))*.6+6)))
    palette={"answer_first":"#1f77b4","bullet_first":"#ff7f0e","contextual":"#98df8a","document_heading":"#9467bd",
             "instructional":"#c49c94","narrative":"#7f7f7f","problem_restatement":"#dbdb8d","refusal":"#9edae5"}
    assert set(categories).issubset(palette)
    colors=[palette[c] for c in categories]
    for ax,model in zip(axes.flat,MODEL_ORDER):
        matrix=rows.loc[rows.model_id.eq(model)].pivot(index="persona",columns="opening_category",values="proportion").reindex(index=persona_order(group),columns=categories).fillna(0)
        np.testing.assert_allclose(matrix.sum(axis=1),1,atol=1e-12)
        bottom=np.zeros(len(matrix))
        for category,color in zip(categories,colors):
            val=100*matrix[category].to_numpy()
            ax.barh(range(len(matrix)),val,left=bottom,color=color,label=category)
            bottom+=val
        ax.set_yticks(range(len(matrix)),[persona_label(group,p) for p in matrix.index],fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel("Proporsi pembuka (%)");ax.set_title(MODEL_LABELS[model])
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,bbox_to_anchor=(.5,.015),fontsize=10)
    fig.suptitle("ETHICS · Komposisi pembuka respons")
    fig.subplots_adjust(left=.12,right=.98,bottom=.13,top=.94,hspace=.32,wspace=.3)
    writer.save(fig,"opening",rows,"opening","Setiap respons memiliki satu kategori pembuka; setiap batang berjumlah 100%. Uji Stuart–Maxwell berpasangan dan p Holm berada di tabel opening.csv. Kategori tidak dianggap sebagai outcome biner independen.","opening_proportion")


def sensitivity_plot(d, writer):
    keys=["model_id","comparison","left_persona","right_persona","outcome_id","scale"]
    all_rows=d.loc[d.population.eq("all_visible")]
    committed=d.loc[d.population.eq("both_committed")]
    joined=all_rows.merge(committed,on=keys,suffixes=("_all","_committed"),validate="one_to_one")
    joined["pair_retention"]=joined.n_pairs_committed/joined.n_pairs_all.replace(0,np.nan)
    joined["direction_reversed"]=joined.signed_effect_all*joined.signed_effect_committed<0
    selected=joined.loc[joined.tier_all.eq("core_confirmatory")]
    fig,axes=plt.subplots(3,2,figsize=(13,12),sharex=True,sharey=True)
    for ax,model in zip(axes.flat,MODEL_ORDER):
        panel=selected.loc[selected.model_id.eq(model)]
        for comparison,color in [("vs_neutral","#2864a5"),("vs_explicit","#b84c35"),("direct","#44815e")]:
            part=panel.loc[panel.comparison.eq(comparison)]
            if len(part): ax.scatter(part.signed_effect_all,part.signed_effect_committed,s=12,alpha=.5,color=color,label=comparison)
        ax.plot([-1,1],[-1,1],color="gray",lw=1);ax.axhline(0,color="gray",lw=.5);ax.axvline(0,color="gray",lw=.5)
        ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_title(MODEL_LABELS[model]);ax.set_xlabel("Seluruh teks: rank-biserial");ax.set_ylabel("Keduanya committed")
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=3)
    fig.suptitle("ETHICS · Sensitivitas terhadap status committed")
    fig.subplots_adjust(left=.1,right=.98,top=.93,bottom=.09,wspace=.3,hspace=.4)
    writer.save(fig,"sensitivity",selected,"sensitivity","Setiap titik adalah satu uji marker inti/model/kontras yang tersedia pada kedua populasi. Garis diagonal menunjukkan efek yang sama. Retensi pasangan dan pembalikan arah tersedia di CSV; kesesuaian arah bukan bukti tidak ada selection bias. Kontras tanpa sensitivitas pada sumber tidak diisi.","rank_biserial_all_vs_committed")
    return joined


def old_inventory(notebook, output, group):
    nb=nbformat.read(notebook,as_version=4)
    (output/"notebook_before.ipynb").write_bytes(notebook.read_bytes())
    rows=[]
    for cell in nb.cells:
        for index,item in enumerate(cell.get("outputs",[])):
            encoded=item.get("data",{}).get("image/png")
            if encoded:
                raw=base64.b64decode(encoded)
                rows.append({"cell_id":cell.id,"output_index":index,"image_sha256":hashlib.sha256(raw).hexdigest(),
                             "action":"retained_in_supporting_notebook", "reason":"User requests overall-only primary presentation; original output preserved in full backup"})
    pd.DataFrame(rows).to_csv(output/"previous_image_inventory.csv",index=False)
    return len(rows)


def build_group(group, statistics, output):
    output.mkdir()
    title, notebook_path, _ = prior.CONFIG[group]
    notebook=ROOT/notebook_path
    old_count=old_inventory(notebook,output,group)
    pool,d,frame,specific,opening,index,manifest=load_data(group,statistics)
    d.to_csv(output/"all_comparisons.csv",index=False)
    opening.to_csv(output/"opening.csv",index=False)
    prior.summarize(d.loc[d.population.eq("all_visible")]).to_csv(output/"summary.csv",index=False)
    # Use the tested local explorer, then remove the subset selector and add population.
    prior.explorer(d.loc[d.population.eq("all_visible")],output,title,overall=True)
    writer=FigureWriter(output)
    coverage_plot(frame,writer,group)
    verbosity_plot(frame,writer,group)
    effect_maps(d,writer,group)
    specific_direct_maps(d,specific,writer)
    specific_prevalence(frame,specific,writer,group).to_csv(output/"specific_descriptives.csv",index=False)
    opening_plot(frame,writer,group)
    sensitivity_plot(d,writer).to_csv(output/"sensitivity.csv",index=False)
    pd.DataFrame(writer.index).to_csv(output/"figure_index.csv",index=False)
    shutil.copyfile(pool/"subset_weights.csv",output/"subset_weights.csv")
    shutil.copyfile(pool/"pair_composition.csv",output/"pair_composition.csv")
    expected=manifest.get("rows")
    assert len(frame)==expected
    source_files={str(pool/name):value for name,value in index.items()}
    if group=="gender": source_files[str(GENDER_MARKERS)]=sha256(GENDER_MARKERS)
    write_json(output/"source_manifest.json",{"result_status":"candidate","source_bundle":str(pool),"source_hashes":source_files,
        "bootstrap_resamples":manifest["bootstrap_resamples"],"source_responses":len(frame),"all_visible_test_rows":int(d.population.eq("all_visible").sum()),
        "aggregation":"response_weighted_ethics_all","renderer_sha256":sha256(Path(__file__)),"previous_image_outputs":old_count,
        "new_figures":len(writer.index),"ci_estimand":"median paired delta (numeric/density); prevalence delta (presence)",
        "notebook_target":str(notebook.relative_to(ROOT)),"notebook_before_sha256":sha256(notebook)})
    print(f"[{group}] Rendered {len(writer.index)} overall figures; {old_count} previous image outputs retained in backup",flush=True)


def revise_group(group, statistics, previous, output):
    """Carry unchanged audited artifacts forward and redraw only visual fixes."""
    output.mkdir()
    prior_manifest=json.loads((previous/"source_manifest.json").read_text(encoding="utf-8"))
    index=pd.read_csv(previous/"figure_index.csv").fillna("")
    validation=json.loads((previous/"audit/numeric_validation.json").read_text(encoding="utf-8"))
    assert validation["valid"]
    audits=pd.read_csv(previous/"audit/numeric_audit.csv").set_index("figure")
    for path in previous.iterdir():
        if path.is_file() and path.name not in {"figure_index.csv","source_manifest.json"}:
            shutil.copyfile(path,output/path.name)
    writer=FigureWriter(output)
    carry=index.loc[~index.section.isin(["specific_direct","opening"])]
    for row in carry.to_dict("records"):
        audit=audits.loc[row["figure"]]
        for field,hash_field in [("figure","png_sha256"),("svg","svg_sha256"),("table","table_sha256")]:
            assert sha256(previous/row[field])==audit[hash_field]
            shutil.copyfile(previous/row[field],output/row[field])
        writer.index.append(row)
    _,d,frame,ids,_,_,_=load_data(group,statistics)
    specific_direct_maps(d,ids,writer)
    opening_plot(frame,writer,group)
    updated=pd.DataFrame(writer.index).set_index("figure").loc[index.figure].reset_index()
    assert len(updated)==len(index)
    updated.to_csv(output/"figure_index.csv",index=False)
    write_json(output/"source_manifest.json",{**prior_manifest,"visual_revision_from":str(previous),
        "revised_renderer_sha256":sha256(Path(__file__)),"carried_unchanged_figures":len(carry),
        "visual_fixes":["Separate footer from model tick labels with physical margin and overlap assertion","Use one opening-category palette across all domains"]})
    print(f"[{group}] Revised {len(index)-len(carry)} figures; {len(carry)} carried byte-identically",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--statistics-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--groups",nargs="+",default=list(prior.CONFIG),choices=list(prior.CONFIG))
    parser.add_argument("--revise-from",type=Path)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    assert output.is_relative_to(REPORTS.resolve())
    output.mkdir(parents=True,exist_ok=False)
    apply_style()
    for group in args.groups:
        print(f"[{group}] Preparing overall presentation",flush=True)
        if args.revise_from:
            revise_group(group,args.statistics_dir.resolve(),args.revise_from.resolve()/group,output/group)
        else:
            build_group(group,args.statistics_dir.resolve(),output/group)


if __name__=="__main__":
    main()
