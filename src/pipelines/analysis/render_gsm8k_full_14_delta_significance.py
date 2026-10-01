"""Re-render the 14-model delta figures from the verified output tables."""

from __future__ import annotations

import argparse

from generate_gsm8k_pilot_model_selection_mcnemar import (
    build_aggregate_significance_markers,
    configure_plots,
    find_repo_root,
    plot_exact_mcnemar_pvalue_heatmap,
    plot_full_heatmap,
    plot_legacy_delta_comparison,
    write_latex_snippet,
    write_readme,
)
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pvalue-only",
        action="store_true",
        help="Render only the raw/Holm Exact McNemar p-value heatmap.",
    )
    args = parser.parse_args()
    repo_root = find_repo_root()
    output_dir = (
        repo_root
        / "result"
        / "GMS8K"
        / "reports"
        / "gsm8k_full_1319_delta_exact_mcnemar_14_models"
    )
    pairwise = pd.read_csv(output_dir / "gsm8k_vs_neutral_exact_mcnemar.csv")
    summary = pd.read_csv(output_dir / "gsm8k_granular_model_summary.csv")
    aggregate = pd.read_csv(output_dir / "gsm8k_legacy_delta_definitions.csv")
    pairwise["significant_holm_0_05"] = (
        pairwise["significant_holm_0_05"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    )
    markers = build_aggregate_significance_markers(aggregate, pairwise)
    markers.to_csv(
        output_dir / "gsm8k_aggregate_delta_significance_markers.csv", index=False
    )

    configure_plots()
    if args.pvalue_only:
        plot_exact_mcnemar_pvalue_heatmap(pairwise, summary, output_dir)
        print(f"Updated p-value heatmap in {output_dir}")
        return

    plot_legacy_delta_comparison(
        aggregate,
        output_dir,
        man_column="delta_man_avg_vs_neutral",
        woman_column="delta_woman_avg_vs_neutral",
        gap_column="gender_gap_man_minus_woman_vs_neutral",
        man_label="Man Avg − Neutral",
        woman_label="Woman Avg − Neutral",
        gap_label="Man Avg − Woman Avg",
        title="GSM8K Gender: Analisis Delta terhadap Persona Neutral",
        subtitle="Rata-rata empat kondisi persona per gender dikurangi akurasi Neutral",
        stem="01_delta_avg_vs_neutral_all_available",
        significance_markers=markers,
    )
    plot_legacy_delta_comparison(
        aggregate,
        output_dir,
        man_column="delta_man_implicit_vs_explicit",
        woman_column="delta_woman_implicit_vs_explicit",
        gap_column="gender_gap_man_minus_woman_imp_vs_exp",
        man_label="Man Imp Avg − Explicit",
        woman_label="Woman Imp Avg − Explicit",
        gap_label="Delta Man − Delta Woman",
        title="GSM8K Gender: Analisis Delta Implicit terhadap Explicit",
        subtitle="Rata-rata tiga kondisi implicit dikurangi kondisi explicit per gender",
        stem="02_delta_implicit_vs_explicit_all_available",
    )
    plot_full_heatmap(pairwise, summary, output_dir)
    plot_exact_mcnemar_pvalue_heatmap(pairwise, summary, output_dir)
    write_latex_snippet(summary, aggregate, output_dir)
    write_readme(output_dir)

    expected = {
        ("allenai_olmo-3-32b-think", "Man Avg − Neutral"),
        ("allenai_olmo-3-32b-think", "Woman Avg − Neutral"),
        ("deepseek_deepseek-r1-distill-qwen-32b", "Man Avg − Neutral"),
        ("deepseek_deepseek-r1-distill-qwen-32b", "Woman Avg − Neutral"),
    }
    actual = set(
        markers.loc[
            markers["star_marker"].eq("*"), ["model", "aggregate_bar"]
        ].itertuples(index=False, name=None)
    )
    assert actual == expected, (actual, expected)
    print(f"Updated figures and marker audit in {output_dir}")


if __name__ == "__main__":
    main()
