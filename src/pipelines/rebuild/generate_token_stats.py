"""Generate token/length/accuracy stats per persona and per explicit-implicit group."""
import sys, re, json, csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ROOT   = Path("result/benchmarks/ethics/runs/deepseek_deepseek-r1-distill-qwen-32b/v9_pilot")
CSV_IN = Path("result/benchmarks/ethics/runs/deepseek_deepseek-r1-distill-qwen-32b/result_ethics_v9_pilot_deepseek_deepseek-r1-distill-qwen-32b.csv")
OUT_DIR = Path("result/benchmarks/ethics/runs/deepseek_deepseek-r1-distill-qwen-32b")

df_csv = pd.read_csv(CSV_IN)

# ── Persona metadata ───────────────────────────────────────────────────────────
def parse_persona(p):
    """Return (dimension, group, mode, variant) for a persona string."""
    if p == "neutral":
        return "neutral", "neutral", "neutral", None

    # detect mode
    if "_explicit" in p:
        mode = "explicit"
        variant = None
    else:
        m = re.search(r"_implicit_(\d+)$", p)
        mode = "implicit"
        variant = int(m.group(1)) if m else None

    # detect dimension + group
    for group in ["man", "woman"]:
        if p.startswith(group + "_"):
            return "gender", group, mode, variant
    for group in ["islam", "christian", "jewish", "atheist"]:
        if p.startswith(group + "_"):
            return "religion", group, mode, variant
    for group in ["african", "east_asian", "south_asian", "western"]:
        if p.startswith(group + "_"):
            return "regional", group, mode, variant
    return "unknown", p, mode, variant

# ── Load all JSON files and extract per-question stats ────────────────────────
# Build lookup: id → (parse_source, is_correct, ground_truth_label, prediction_label)
# Use rebuilt CSV as ground truth for scored/is_correct
csv_lookup = {}
for _, row in df_csv.iterrows():
    csv_lookup[(row["id"], row["persona"])] = {
        "parse_source":       row["parse_source"],
        "is_correct":         row["is_correct"],   # True/False/"" from rebuilt CSV
        "ground_truth_label": row["ground_truth_label"],
        "prediction_label":   row["prediction_label"],
    }

print("Loading JSON files for token data...")
records = []

for jf in sorted(ROOT.rglob("*.json")):
    if "warmup" in jf.parts:
        continue
    try:
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        continue

    persona = d.get("persona", "")
    subset  = d.get("subset", "")
    qid     = d.get("id", "")
    pred_d  = d.get("prediction", {}) or {}
    resp_d  = d.get("response", {}) or {}

    # use rebuilt CSV for is_correct / parse_source (key = id + persona)
    csv_row      = csv_lookup.get((qid, persona), {})
    parse_source = csv_row.get("parse_source", "none")
    is_correct   = csv_row.get("is_correct", "")     # True / False / ""
    gt           = csv_row.get("ground_truth_label")
    scored       = parse_source != "none"

    # tokens from raw API response
    raw_api = resp_d.get("raw_api_response", {}) or {}
    usage   = raw_api.get("usage", {}) or {}
    prompt_tok     = usage.get("prompt_tokens", np.nan)
    completion_tok = usage.get("completion_tokens", np.nan)
    total_tok      = usage.get("total_tokens", np.nan)
    details        = usage.get("completion_tokens_details", {}) or {}
    reasoning_tok  = details.get("reasoning_tokens", np.nan)
    cost           = usage.get("cost", np.nan)

    # response text stats
    raw_text   = str(pred_d.get("raw", "") or "").strip()
    word_count = len(raw_text.split()) if raw_text else 0
    char_count = len(raw_text)
    latency    = resp_d.get("latency_sec", np.nan)

    dimension, group, mode, variant = parse_persona(persona)

    records.append({
        "persona":            persona,
        "dimension":          dimension,
        "group":              group,
        "mode":               mode,
        "variant":            variant,
        "subset":             subset,
        "id":                 qid,
        "gt":                 gt,
        "is_correct":         is_correct,
        "scored":             scored,
        "parse_source":       parse_source,
        "prompt_tokens":      prompt_tok,
        "completion_tokens":  completion_tok,
        "reasoning_tokens":   reasoning_tok,
        "total_tokens":       total_tok,
        "cost":               cost,
        "word_count":         word_count,
        "char_count":         char_count,
        "latency_sec":        latency,
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} question records across {df['persona'].nunique()} personas\n")

# ── Helper: aggregate stats for a group of rows ───────────────────────────────
def agg_stats(rows):
    n_total  = len(rows)
    n_scored = rows["scored"].sum()
    n_none   = (rows["parse_source"] == "none").sum()
    correct_rows = rows[rows["is_correct"] == True]
    n_correct = len(correct_rows)
    accuracy  = n_correct / n_scored if n_scored > 0 else np.nan
    none_rate = n_none / n_total if n_total > 0 else np.nan

    def m(col): return round(rows[col].mean(), 2)
    def s(col): return round(rows[col].std(), 2)

    return {
        "n_total":   n_total,
        "n_scored":  int(n_scored),
        "n_none":    int(n_none),
        "none_rate": round(none_rate, 4),
        "n_correct": int(n_correct),
        "accuracy":  round(accuracy, 4) if not np.isnan(accuracy) else "",
        "mean_prompt_tokens":     m("prompt_tokens"),
        "std_prompt_tokens":      s("prompt_tokens"),
        "mean_completion_tokens": m("completion_tokens"),
        "std_completion_tokens":  s("completion_tokens"),
        "mean_reasoning_tokens":  m("reasoning_tokens"),
        "std_reasoning_tokens":   s("reasoning_tokens"),
        "mean_total_tokens":      m("total_tokens"),
        "std_total_tokens":       s("total_tokens"),
        "mean_word_count":        m("word_count"),
        "std_word_count":         s("word_count"),
        "mean_char_count":        m("char_count"),
        "std_char_count":         s("char_count"),
        "mean_latency_sec":       m("latency_sec"),
        "mean_cost":              round(rows["cost"].mean(), 6),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CSV 1: Per Persona (41 rows)
# ═══════════════════════════════════════════════════════════════════════════════
rows_per_persona = []
for persona, grp in sorted(df.groupby("persona")):
    dim, group, mode, variant = parse_persona(persona)
    stats = agg_stats(grp)
    row = {
        "persona":    persona,
        "dimension":  dim,
        "group":      group,
        "mode":       mode,
        "variant":    variant if variant is not None else "",
        **stats,
    }
    rows_per_persona.append(row)

df_persona = pd.DataFrame(rows_per_persona)
out_persona = OUT_DIR / "token_stats_per_persona.csv"
df_persona.to_csv(out_persona, index=False, encoding="utf-8-sig")
print(f"[OK] token_stats_per_persona.csv — {len(df_persona)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# CSV 2: By Explicit / Implicit / Neutral
# ═══════════════════════════════════════════════════════════════════════════════
rows_mode = []

def add_mode_row(label_dict, subset_df):
    if len(subset_df) == 0:
        return
    stats = agg_stats(subset_df)
    rows_mode.append({**label_dict, **stats})

# Overall
add_mode_row({"group_label": "OVERALL", "dimension": "all", "mode": "all", "variant": ""}, df)

# Neutral
add_mode_row({"group_label": "neutral", "dimension": "neutral", "mode": "neutral", "variant": ""}, df[df["mode"] == "neutral"])

# All explicit vs all implicit
add_mode_row({"group_label": "ALL_explicit", "dimension": "all", "mode": "explicit", "variant": ""}, df[df["mode"] == "explicit"])
add_mode_row({"group_label": "ALL_implicit", "dimension": "all", "mode": "implicit", "variant": ""}, df[df["mode"] == "implicit"])
for v in [1, 2, 3]:
    add_mode_row({"group_label": f"ALL_implicit_{v}", "dimension": "all", "mode": "implicit", "variant": str(v)},
                 df[(df["mode"] == "implicit") & (df["variant"] == v)])

# Per dimension × mode
for dim in ["gender", "religion", "regional"]:
    sub = df[df["dimension"] == dim]
    add_mode_row({"group_label": f"{dim}_all",      "dimension": dim, "mode": "all",      "variant": ""}, sub)
    add_mode_row({"group_label": f"{dim}_explicit",  "dimension": dim, "mode": "explicit", "variant": ""}, sub[sub["mode"] == "explicit"])
    add_mode_row({"group_label": f"{dim}_implicit",  "dimension": dim, "mode": "implicit", "variant": ""}, sub[sub["mode"] == "implicit"])
    for v in [1, 2, 3]:
        add_mode_row({"group_label": f"{dim}_implicit_{v}", "dimension": dim, "mode": "implicit", "variant": str(v)},
                     sub[(sub["mode"] == "implicit") & (sub["variant"] == v)])

# Per individual group × mode (man_explicit, man_implicit, etc.)
for (dim, group, mode), grp in df[df["mode"] != "neutral"].groupby(["dimension", "group", "mode"]):
    add_mode_row({"group_label": f"{group}_{mode}", "dimension": dim, "mode": mode, "variant": ""},  grp)
    if mode == "implicit":
        for v in [1, 2, 3]:
            vgrp = grp[grp["variant"] == v]
            if len(vgrp):
                add_mode_row({"group_label": f"{group}_implicit_{v}", "dimension": dim, "mode": mode, "variant": str(v)}, vgrp)

df_mode = pd.DataFrame(rows_mode)
out_mode = OUT_DIR / "token_stats_by_mode.csv"
df_mode.to_csv(out_mode, index=False, encoding="utf-8-sig")
print(f"[OK] token_stats_by_mode.csv — {len(df_mode)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# CSV 3: Per Subset × Mode (breakdown per ETHICS subset)
# ═══════════════════════════════════════════════════════════════════════════════
rows_subset = []
for (subset, mode), grp in df.groupby(["subset", "mode"]):
    stats = agg_stats(grp)
    rows_subset.append({"subset": subset, "mode": mode, **stats})

df_subset = pd.DataFrame(rows_subset)
out_subset = OUT_DIR / "token_stats_by_subset_mode.csv"
df_subset.to_csv(out_subset, index=False, encoding="utf-8-sig")
print(f"[OK] token_stats_by_subset_mode.csv — {len(df_subset)} rows")

# ── Quick terminal preview ─────────────────────────────────────────────────────
print("\n=== PREVIEW: token_stats_by_mode (key cols) ===")
cols = ["group_label", "n_total", "n_scored", "none_rate", "accuracy",
        "mean_prompt_tokens", "mean_completion_tokens", "mean_reasoning_tokens",
        "mean_word_count", "mean_latency_sec"]
print(df_mode[cols].to_string(index=False))
