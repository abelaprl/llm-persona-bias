# Candidate Bab IV summary: gender-lm-v1.2.1-candidate

Status: **candidate**. Do not treat these tables as final thesis findings until
`audit_marker_hits.csv` and `audit_recall_responses.csv` have been independently
annotated and agreement has been reported.

The executed analysis notebook is
`D:\TugasAkhir\notebooks\linguistic_marker_gender_gsm8k_v1_2_1.ipynb`.
Its visual outputs are:

- `images/01_text_coverage_by_model.png`
- `images/02_explicit_vs_neutral_marker_presence_heatmap.png`
- `images/03_primary_numeric_significance_count_heatmap.png`
- `images/04_explicit_vs_neutral_opening_heatmap.png`
- `images/05_contrastive_wordcloud_personas_vs_neutral.png`
- `images/06_contrastive_wordcloud_implicit_vs_explicit.png`

The contrastive word clouds are not raw-frequency clouds. Token presence is
paired by model and question, persona-minus-comparator prevalence is calculated
per model, and model deltas receive equal weight. Displayed tokens have the same
direction in at least four of six models. Math-domain boilerplate and LaTeX
serialization terms are excluded. Reproducible values and parameters are in:

- `lexical_contrast_token_weights.csv`
- `lexical_contrast_metadata.json`

Word size is descriptive effect size, not a p-value.

These figures are inspection aids and candidate Bab IV material; they do not
replace the manual precision and recall audit.

This bundle is derived from `D:\TugasAkhir\result\linguistic_markers\gender_gsm8k_full_source_aligned_v1_2_1_20260817`. Correctness was not used to select
responses or contrasts. Files whose names contain `holm_significant` are views
filtered at `p_holm < 0.05`; every planned contrast, including non-significant
ones, remains in the source run's paired tables.

Row counts:

```json
{
  "numeric_planned": 1596,
  "numeric_primary_holm_significant": 527,
  "numeric_supplementary_holm_significant": 85,
  "presence_planned": 1428,
  "presence_holm_significant": 439,
  "opening_planned": 924,
  "opening_holm_significant": 147,
  "omnibus_planned": 372,
  "omnibus_holm_significant": 157
}
```
