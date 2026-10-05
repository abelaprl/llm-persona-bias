# Gender GSM8K linguistic-marker visual revision v2

Status: **candidate pending manual precision/recall audit**.

Source run: `D:\TugasAkhir\result\linguistic_markers\gender_gsm8k_full_source_aligned_v1_2_1_20260817`  
Source records: 71,215  
Bootstrap resamples: 10,000

## Methodological decisions

- The primary analysis still includes correct and incorrect responses.
- Correctness sensitivity is paired first, then restricted to questions where
  both compared conditions are correct.
- E, I1, I2, and I3 are never averaged into one inferential condition.
- Numeric figures lead with rank-biserial effect size; p Holm is the vertical
  axis, not the visual selection rule.
- Forest plots report paired prevalence differences with 95% bootstrap CIs.
- Opening categories are visualized as 100% compositions and tested jointly
  with Stuart-Maxwell marginal homogeneity.
- Ranked lexical lollipops replace word clouds and retain exact delta values.
- Marker extraction uses visible final-answer text only; hidden reasoning is
  metadata and never a fallback text surface.

The two manual audit CSVs in the source run are not complete unless
`manual_audit_status.csv` reports annotated rows.  Do not promote candidate
claims to final Bab IV findings before precision, recall, and agreement are
reported.


## Dedicated gendered-word analysis

The open-vocabulary lollipop analysis removes ordinary pronouns as stopwords.
The dedicated `gendered-visible-lexicon-v1` analysis therefore measures `he/him/his`,
`she/her/hers`, and a closed list of human-denoting gender nouns separately.

- Figure 11 is the requested gender-word cloud. Its size is an equal-model mean
  of occurrences per 10,000 visible-response words.
- Figure 12 shows exact pronoun-class response prevalence for every model and
  all nine personas.
- Figure 13 reports paired I1/I2/I3-minus-neutral prevalence differences with
  95% bootstrap confidence intervals.

Prompt text and hidden reasoning are not included. Exact counts and rates are
stored in the four `gender*csv` tables.
