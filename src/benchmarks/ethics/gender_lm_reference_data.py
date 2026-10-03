"""Read-only adapters for the reference-based Gender ETHICS presentation."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "result/benchmarks/ethics/reports/gender_lm_v2_candidate_20260916_r3"
DIAGNOSTICS = ROOT / "result/benchmarks/ethics/reports/gender_lm_diagnostics_20260919_r1"
ACTIVE = ROOT / "notebooks/gender/linguistic marker/ETHICS"
NOTEBOOK = "ETHICS_Gender_Linguistic_Markers_Analysis_v1_5_candidate.ipynb"
MODELS = ["meta-llama_llama-3.1-8b-instruct", "meta-llama_llama-3.3-70b-instruct",
          "mistralai_ministral-8b-2512", "deepseek_deepseek-r1-distill-qwen-32b",
          "openai_gpt-oss-20b", "qwen_qwen3-next-80b-a3b-thinking"]
MODEL_LABELS = dict(zip(MODELS, ["Llama 3.1 8B", "Llama 3.3 70B", "Ministral 8B",
                               "DeepSeek R1 32B", "GPT-OSS 20B", "Qwen3 Next 80B"]))
COLORS = dict(zip(MODELS, ["#2563EB", "#7C3AED", "#059669", "#DC2626", "#D97706", "#0891B2"]))
SCENARIOS = ["E", "I1", "I2", "I3"]
PERSONAS = ["neutral"] + [f"{g}_{s}" for s in ["explicit_1", "implicit_1", "implicit_2", "implicit_3"]
                           for g in ["man", "woman"]]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scenario(persona):
    if persona == "neutral":
        return "N"
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    return "E" if "explicit" in persona else "I" + persona[-1]


def persona(gender, condition):
    if condition == "N":
        return "neutral"
    return gender.lower() + ("_explicit_1" if condition == "E" else "_implicit_" + condition[-1])


def select_contrasts(comparisons, marker_ids, comparison, measurement):
    """Keep source orientation, statistics and units; add display-only columns."""
    data = comparisons.loc[comparisons.scope.eq("overall") & comparisons.marker_id.isin(marker_ids)
                           & comparisons.measurement.eq(measurement)].copy()
    if not data.effect_direction.eq("left_minus_right").all():
        raise ValueError("Unexpected source direction")
    left_scenario = data.left.map(scenario)
    right_scenario = data.right.map(scenario)
    if comparison == "direct":
        mask = data.left.str.startswith("woman_") & data.right.str.startswith("man_") & left_scenario.eq(right_scenario)
    elif comparison == "neutral":
        mask = data.right.eq("neutral") & data.left.ne("neutral")
    elif comparison == "implicit":
        mask = left_scenario.isin(["I1", "I2", "I3"]) & right_scenario.eq("E")
        mask &= data.left.str.split("_").str[0].eq(data.right.str.split("_").str[0])
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    data = data.loc[mask].copy()
    data["panel_gender"] = "Woman − Man" if comparison == "direct" else data.left.str.split("_").str[0].str.title()
    data["panel_scenario"] = data.left.map(scenario)
    keys = ["model_id", "marker_id", "measurement", "panel_gender", "panel_scenario"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate comparison")
    if measurement == "density":
        data["plot_value"] = data.matched_rank_biserial
        data["plot_ci_low"] = np.nan
        data["plot_ci_high"] = np.nan
        data["plot_unit"] = "matched_rank_biserial"
    elif measurement == "presence":
        data["plot_value"] = data.effect
        data["plot_ci_low"] = data.ci_low
        data["plot_ci_high"] = data.ci_high
        data["plot_unit"] = "percentage_points"
    else:
        raise ValueError("Unsupported plot measurement")
    return data


def category_diversity(responses, marker_ids):
    """Count distinct dictionary categories, not occurrences or semantic richness."""
    out = responses.copy()
    counts = out[[m + "__count" for m in marker_ids]]
    available = out.text_available.eq(True)
    if counts.loc[available].isna().any().any():
        raise ValueError("Available response has missing marker counts")
    out["category_count"] = counts.gt(0).sum(axis=1).astype(float).where(available)
    return out


def profiles(summary):
    """Join prevalence with raw occurrence and density data without pooling models."""
    data = summary.loc[summary.scope.eq("overall")].copy()
    keys = ["scope", "model_id", "persona", "marker_id"]
    out = data.loc[data.measurement.eq("presence")].copy()
    density = data.loc[data.measurement.eq("density"), keys + ["value"]].rename(columns={"value": "density_per_100_words"})
    return out.merge(density, on=keys, how="left", validate="one_to_one")
