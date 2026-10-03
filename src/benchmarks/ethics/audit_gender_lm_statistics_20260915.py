"""Read-only independent paired-statistic audit; writes only this new bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description="Recompute frozen Gender ETHICS LM paired statistics independently.")
parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for audit artifacts")
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
SOURCE = ROOT / "result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers_gender_candidate_20260821_v1_6_0/response_level_markers.csv.gz"
BUNDLE = ROOT / "result/benchmarks/ethics/reports/Gender_Persona/linguistic_markers/analysis/gender_ethics_lm_gabungan_20260909_r4"


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


tables = []
for kind in ("numeric", "gender_density", "gender_presence"):
    for mode in ("all_text", "committed"):
        for contrast in ("vs_neutral", "vs_explicit", "direct"):
            name = f"{kind}_{contrast}_{mode}.csv"
            table = pd.read_csv(BUNDLE / name)
            table["source_file"] = name
            table["source_line"] = np.arange(2, len(table) + 2)
            table["mode"] = mode
            table["kind"] = kind
            table["scope"] = "direct" if contrast == "direct" else "within"
            tables.append(table)
all_rows = pd.concat(tables, ignore_index=True)
outcomes = set(all_rows.outcome_id)
surface = {"word_count", "avg_sentence_length", "ttr_300"}
usecols = ["model_id", "persona", "subset", "question_id", "answer_status", "text_available"]
for outcome in outcomes:
    if outcome in surface:
        usecols.append(outcome)
    else:
        usecols.extend([outcome + "__density", outcome + "__present"])
frame = pd.read_csv(SOURCE, usecols=list(set(usecols)))
keys = ["model_id", "persona", "subset", "question_id"]
assert not frame.duplicated(keys).any()
indexed = {(m, p): g.set_index(["subset", "question_id"]) for (m, p), g in frame.groupby(["model_id", "persona"])}
rows = []
for i, row in all_rows.iterrows():
    outcome = row.outcome_id
    column = outcome if outcome in surface else outcome + ("__present" if row.kind == "gender_presence" else "__density")
    left = indexed[(row.model_id, row.left_persona)]
    right = indexed[(row.model_id, row.right_persona)]
    if row["mode"] == "committed":
        left = left[left.answer_status.eq("committed") & left.text_available]
        right = right[right.answer_status.eq("committed") & right.text_available]
    pairs = pd.concat([left[column].rename("left"), right[column].rename("right")], axis=1, join="inner").dropna()
    delta = (pairs.left - pairs.right).to_numpy(float)
    nonzero = delta[delta != 0]
    positive = int(sum(nonzero > 0))
    negative = int(sum(nonzero < 0))
    sign_p = binomtest(positive, len(nonzero)).pvalue if len(nonzero) else 1.0
    r = 0.0
    wp = 1.0
    if len(nonzero):
        rank = rankdata(abs(nonzero))
        r = float((rank[nonzero > 0].sum() - rank[nonzero < 0].sum()) / rank.sum())
        if row.kind != "gender_presence":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wp = float(wilcoxon(delta, zero_method="wilcox", correction=False, alternative="two-sided", method="auto").pvalue)
    skew = np.nan
    if len(nonzero) >= 4:
        q1, med, q3 = np.quantile(nonzero, [.25, .5, .75])
        skew = float((q3 + q1 - 2 * med) / (q3 - q1)) if q3 != q1 else 0.0
    sparse = len(nonzero) / len(delta) < .5 if len(delta) else False
    method = "no_change" if not len(nonzero) else "sign_test" if sparse or abs(skew) > .5 else "wilcoxon_signed_rank"
    recomputed_p = sign_p if row.kind == "gender_presence" or method == "sign_test" else wp
    result = row.to_dict()
    result.update(recomputed_pairs=len(delta), recomputed_nonzero=len(nonzero), recomputed_sign_p=sign_p,
                  recomputed_wilcoxon_p=wp, recomputed_rank_biserial=r, recomputed_p=recomputed_p,
                  pair_count_matches=len(delta) == row.n_pairs,
                  p_error=abs(recomputed_p-row.p_raw),
                  effect_error=abs((float(np.mean(delta)) if row.kind == "gender_presence" else r) - (row.prevalence_delta if row.kind == "gender_presence" else row.rank_biserial)))
    rows.append(result)
    if (i + 1) % 1000 == 0:
        print(f"Recomputed {i+1}/{len(all_rows)}", flush=True)
result = pd.DataFrame(rows)
family = ["kind", "mode", "scope", "model_id", "identity", "outcome_id"]
for _, group in result.groupby(family, dropna=False):
    result.loc[group.index, "recomputed_holm"] = holm(group.recomputed_p)
    result.loc[group.index, "wilcoxon_primary_holm"] = holm(group.recomputed_wilcoxon_p)
summary = []
for (kind, mode), group in result.groupby(["kind", "mode"]):
    current = group.p_holm < .05
    alternative = group.wilcoxon_primary_holm < .05
    item = dict(kind=kind, mode=mode, rows=len(group), max_p_error=group.p_error.max(),
                max_effect_error=group.effect_error.max(), pairs_mismatched=int((~group.pair_count_matches).sum()),
                max_holm_error=float(abs(group.recomputed_holm-group.p_holm).max()),
                significant_current=int(current.sum()))
    if kind != "gender_presence":
        item.update(significant_wilcoxon=int(alternative.sum()), lost_with_wilcoxon=int((current & ~alternative).sum()),
                    gained_with_wilcoxon=int((~current & alternative).sum()),
                    test_counts=group.test_used.value_counts().to_dict(),
                    sign_rank_direction_disagreement=int(((group.test_used == "sign_test") & ((group.n_positive-group.n_negative)*group.rank_biserial<0)).sum()))
    summary.append(item)
result.to_csv(OUT / "independent_recomputation.csv", index=False, mode="x")
pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False, mode="x")
numeric_visible = result[(result.kind != "gender_presence") & result["mode"].eq("all_text")]
disagreement = numeric_visible[
    numeric_visible.test_used.eq("sign_test")
    & ((numeric_visible.n_positive - numeric_visible.n_negative) * numeric_visible.rank_biserial < 0)
]
disagreement.to_csv(OUT / "sign_rank_direction_disagreement.csv", index=False, mode="x")
disagreement[disagreement.p_holm < .05].to_csv(
    OUT / "significant_direction_disagreement.csv", index=False, mode="x"
)
metadata = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), source_rows=len(frame),
                source_unique_keys=len(frame[keys].drop_duplicates()), summaries=summary,
                limits=["No marker extraction or manual semantic validation performed", "Bootstrap confidence intervals and opening multinomial tests not recomputed", "Wilcoxon alternative is sensitivity analysis, not authorization to replace active results"])
with (OUT / "summary.json").open("x", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)
print(json.dumps(metadata, indent=2, ensure_ascii=False))
