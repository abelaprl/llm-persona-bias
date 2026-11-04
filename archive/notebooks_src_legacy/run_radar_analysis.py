#!/usr/bin/env python
"""
Execute 5-group, 3-group radar charts and summary statistics.
This script imports the notebook environment and runs additional analysis.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up environment similar to notebook
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 200)

# Load the data (mirror what notebook does)
DATA_PATH = Path("results_all_models_all_personas.csv") if Path("results_all_models_all_personas.csv").exists() else \
            Path(r"E:\Tugas Akhir\TugasAkhir\result\reports\results_all_models_all_personas.csv")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns\n")

# Preprocessing (copy from notebook)
persona_order = [
    "man_implicit_1", "man_implicit_2", "man_implicit_3",
    "woman_implicit_1", "woman_implicit_2", "woman_implicit_3",
    "man_explicit_1", "woman_explicit_1", "neutral",
]

def normalize_is_correct(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan

df["response_content_full"] = df["response_content_full"].fillna("").astype(str)
df["is_correct"] = df["is_correct"].apply(normalize_is_correct)
df = df[df["persona_folder"].isin(persona_order)].copy()
df["persona_folder"] = pd.Categorical(df["persona_folder"], categories=persona_order, ordered=True)

# Create persona groups (copy functions from notebook)
def create_persona_groups_5(persona_folder):
    text = str(persona_folder).lower() if pd.notna(persona_folder) else ""
    if "neutral" in text:
        return "neutral"
    elif "woman" in text or "women" in text:
        if "explicit" in text:
            return "woman_explicit_avg"
        else:
            return "woman_implicit_avg"
    elif "man" in text or "men" in text:
        if "explicit" in text:
            return "man_explicit_avg"
        else:
            return "man_implicit_avg"
    return None

def create_persona_groups_3(persona_folder):
    text = str(persona_folder).lower() if pd.notna(persona_folder) else ""
    if "neutral" in text:
        return "neutral"
    elif "woman" in text or "women" in text:
        return "woman"
    elif "man" in text or "men" in text:
        return "man"
    return None

df["persona_group_5"] = df["persona_folder"].apply(create_persona_groups_5)
df["persona_group_3"] = df["persona_folder"].apply(create_persona_groups_3)

# Define marker categories (assumed from notebook context)
marker_categories = [
    "sequential", "instructional", "structural", "logical",
    "formal_logic", "verification", "hedging", "boosters",
    "biber_technical", "self_mentions", "engagement", "attitude", "clarification",
]

density_metrics = [f"{cat}_density" for cat in marker_categories]

# Aggregate function
def aggregate_by_persona_groups(data, persona_group_col, metric_cols, filter_correct=True):
    if filter_correct:
        filtered = data[data["is_correct"] == True].copy()
        if filtered.empty:
            filtered = data.copy()
    else:
        filtered = data.copy()
    
    aggregated = (
        filtered.groupby(["model_name", persona_group_col], as_index=False)[metric_cols]
        .mean()
        .sort_values(["model_name", persona_group_col])
    )
    return aggregated

# Check if density columns exist
missing_cols = [col for col in density_metrics if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing density columns: {missing_cols}")
    print(f"Available columns: {[c for c in df.columns if 'density' in c]}\n")
    # Try to create them if they don't exist
    print("Attempting to extract and calculate densities...")
    
    # Just use available density columns
    density_metrics = [c for c in df.columns if c.endswith("_density")]
    if not density_metrics:
        print("ERROR: No density columns found in data!")
        sys.exit(1)
    print(f"Using {len(density_metrics)} available density columns")

# Create aggregates
print(f"Creating aggregates from {len(df):,} rows...")
df_agg_5 = aggregate_by_persona_groups(df, "persona_group_5", density_metrics)
df_agg_3 = aggregate_by_persona_groups(df, "persona_group_3", density_metrics)

print(f"5-Group aggregated: {df_agg_5.shape}")
print(f"3-Group aggregated: {df_agg_3.shape}\n")

# Define radar plotting function
def plot_aggregated_radar_charts(data, persona_group_col, metrics, persona_groups_list, title_suffix=""):
    """Plot aggregated radar charts."""
    if len(persona_groups_list) == 5:
        colors = {
            "man_explicit_avg": "#1f77b4",
            "man_implicit_avg": "#9ecae1",
            "neutral": "#2ca02c",
            "woman_implicit_avg": "#ff7f0e",
            "woman_explicit_avg": "#d62728",
        }
    else:
        colors = {"man": "#1f77b4", "neutral": "#2ca02c", "woman": "#ff7f0e"}
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    global_max = data[metrics].max().max() if not data.empty else 1.0
    radial_max = max(1.0, float(global_max) * 1.15)
    
    model_list = sorted(data["model_name"].unique().tolist())
    models_per_page = 4
    
    print(f"Generating radar charts for {len(model_list)} models ({len(persona_groups_list)} groups per model)...")
    
    for start in range(0, len(model_list), models_per_page):
        page_models = model_list[start : start + models_per_page]
        fig, axes = plt.subplots(2, 2, figsize=(22, 18), subplot_kw={"polar": True}, constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        
        for i, model in enumerate(page_models):
            ax = axes[i]
            model_df = data[data["model_name"] == model]
            
            for persona_group in persona_groups_list:
                row = model_df[model_df[persona_group_col] == persona_group]
                values = row.iloc[0][metrics].tolist() if not row.empty else [0.0] * len(metrics)
                values += values[:1]
                color = colors.get(persona_group, "#333333")
                ax.plot(angles, values, linewidth=2.3, label=persona_group, color=color)
                ax.fill(angles, values, alpha=0.14, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace("_density", "") for m in metrics], fontsize=10)
            ax.set_ylim(0, radial_max)
            ax.set_title(model, fontsize=12, pad=16)
            ax.grid(alpha=0.35)
        
        for j in range(len(page_models), len(axes)):
            fig.delaxes(axes[j])
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(persona_groups_list), frameon=True, fontsize=11)
        
        page_no = (start // models_per_page) + 1
        total_pages = int(np.ceil(len(model_list) / models_per_page))
        fig.suptitle(f"Aggregated Radar Charts {title_suffix}\n(Page {page_no}/{total_pages})", fontsize=16, y=1.02)
        plt.savefig(f"radar_page_{persona_groups_list[0]}_p{page_no}.png", dpi=100, bbox_inches='tight')
        print(f"  Saved: radar_page_{persona_groups_list[0]}_p{page_no}.png")
        plt.close()

# Execute 5-group radar
print("\n" + "="*80)
print("5-GROUP RADAR CHARTS (Gender x Explicit/Implicit)")  
print("="*80)
plot_aggregated_radar_charts(
    df_agg_5,
    persona_group_col="persona_group_5",
    metrics=density_metrics,
    persona_groups_list=["man_explicit_avg", "man_implicit_avg", "neutral", "woman_implicit_avg", "woman_explicit_avg"],
    title_suffix="(Gender x Explicit/Implicit)"
)

# Execute 3-group radar
print("\n" + "="*80)
print("3-GROUP RADAR CHARTS (Gender Only)")
print("="*80)
plot_aggregated_radar_charts(
    df_agg_3,
    persona_group_col="persona_group_3",
    metrics=density_metrics,
    persona_groups_list=["woman", "neutral", "man"],
    title_suffix="(Gender)"
)

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n--- 5-GROUP SETUP (Gender x Explicit/Implicit) ---\n")
summary_5 = df_agg_5.groupby("persona_group_5")[density_metrics].mean()
print("Mean Density (%) per Persona Group:")
print(summary_5.round(3))

print("\n\nTop 5 Markers per Persona Group (5-Group Setup):")
for persona_group in ["man_explicit_avg", "man_implicit_avg", "neutral", "woman_implicit_avg", "woman_explicit_avg"]:
    if persona_group in summary_5.index:
        group_data = summary_5.loc[persona_group]
        top5 = group_data.nlargest(5)
        print(f"\n{persona_group.upper()}:")
        for marker, value in top5.items():
            marker_clean = marker.replace("_density", "")
            print(f"  {marker_clean:25s}: {value:6.3f}%")

print("\n" + "-"*80)
print("\n--- 3-GROUP SETUP (Gender Only) ---\n")
summary_3 = df_agg_3.groupby("persona_group_3")[density_metrics].mean()
print("Mean Density (%) per Persona Group:")
print(summary_3.round(3))

print("\n\nTop 5 Markers per Persona Group (3-Group Setup):")
for persona_group in ["woman", "neutral", "man"]:
    if persona_group in summary_3.index:
        group_data = summary_3.loc[persona_group]
        top5 = group_data.nlargest(5)
        print(f"\n{persona_group.upper()}:")
        for marker, value in top5.items():
            marker_clean = marker.replace("_density", "")
            print(f"  {marker_clean:25s}: {value:6.3f}%")

print("\n" + "="*80)
print("Analysis complete! Check current directory for radar PNG files.")
print("="*80)
