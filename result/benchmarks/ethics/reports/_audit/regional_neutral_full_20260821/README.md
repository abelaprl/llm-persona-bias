# ETHICS Regional + Neutral audit — 2026-08-21

This directory contains a blind, non-destructive audit of:

`result/ETHICS/reports/Regional_Cultural_Persona/_combined/all_models_results_full.csv`

No parser cache, canonical report, score, or response was changed. Ground truth
was not consulted during extraction/adjudication.

## Regional baseline

| status | rows | rate |
|---|---:|---:|
| committed | 105,232 | 72.4020% |
| abstain | 38,428 | 26.4393% |
| missing | 1,684 | 1.1586% |
| total | 145,344 | 100% |

All 38,428 Regional abstentions were scanned using the complete response text.
The scan found 4,538 rows with at least one discovery-only verdict candidate;
178 have opposing candidate directions. Candidate detection is not promotion.

## AITA explicit-label audit

- Full Regional rule universe: 196 hits.
- Deterministic full-response sample: 50.
- Sample adjudication: 49 `ok`, 1 `wrong` (98% observed precision).
- Failure form: a mixed NTA/YTA verdict that assigns different labels to
  different aspects of the situation. Responses that reject YTA/NTA and then
  explicitly select ESH are valid negative AITA verdicts, not false positives.
- Nine baseline abstentions are individually confirmed as genuine final AITA
  verdicts, but the broad rule may not be promoted because its audit was not
  clean. A narrower rule must be sampled again from zero.

## Existing semantic commitments

Regional contains 4,280 rows already committed with confidence
`semantic_unaudited`. Blind reparse leaves all 4,280 as candidates/abstentions:

- `analysis_without_verdict`: 4,254
- `detector_disagreement`: 26

All 26 disagreement rows were reviewed in full. Results:

- 19 baseline semantic decisions matched the response.
- 7 were wrong commitments.
- Of the 19 matching rows, 13 contain an explicit answer form that the current
  explicit resolver missed and can be investigated independently of semantics.

Observed false-positive forms include:

- a neutral outcome forced into `wrong`;
- right and wrong conditional branches forced into one direction;
- an explicit `0.5` forced into a binary answer;
- `the supplied answer 0 is reasonable/correct` harvested as an
  `acceptable` verdict, even though the model selected `0/wrong`.

Because none of the 29 semantic rules has passed its per-rule audit quota, the
4,280 rows cannot be treated as audited semantic commitments.

## Status projections before parser changes

| interpretation | committed | abstain | missing | coverage |
|---|---:|---:|---:|---:|
| current legacy report | 105,232 | 38,428 | 1,684 | 72.4020% |
| legacy + nine individually confirmed AITA rows | 105,241 | 38,419 | 1,684 | 72.4082% |
| conservative: demote all unaudited semantics | 100,952 | 42,708 | 1,684 | 69.4573% |
| conservative + nine AITA rows | 100,961 | 42,699 | 1,684 | 69.4635% |
| conservative + nine AITA + 13 reviewed explicit rows | 100,974 | 42,686 | 1,684 | 69.4724% |

The final three rows are audit projections, not generated reports. Missing is
unchanged because all 1,684 Regional missing rows have no response text.

## Main artifacts

- `noncommitted_response_audit_index_2026-08-16.csv`: exhaustive row-level
  non-committed inventory.
- `noncommitted_response_audit_2026-08-16.json`: full scan summary.
- `parser_refinement_scan/aita_verdict_label_audit_sample_adjudicated.csv`:
  reviewed 50-row AITA sample.
- `parser_refinement_scan/aita_newly_committed_adjudicated.csv`: nine reviewed
  potential recoveries.
- `semantic_commits/regional_semantic_unaudited_full_index.csv`: all 4,280
  Regional semantic commitments with blind raw-parser decisions and full text.
- `semantic_commits/regional_semantic_per_rule_audit_sample.csv`: quota sample
  for every semantic rule (1,541 rows total).
- `semantic_commits/regional_semantic_priority_adjudicated.csv`: reviewed 26
  detector-disagreement rows.

## Implemented conservative refinement

The approved narrow rules were implemented after the discovery audit. The
canonical CSV above remains unchanged. The generated refinement is:

`result/ETHICS/reports/Regional_Cultural_Persona/_combined/parser_refinement_20260821/`

Final Regional-only status:

| status | rows | rate |
|---|---:|---:|
| committed | 101,026 | 69.5082% |
| abstain | 42,634 | 29.3332% |
| missing | 1,684 | 1.1586% |
| total | 145,344 | 100% |

Final Regional + Neutral status is 107,745 committed, 44,837 abstain, and 1,846
missing (69.7704% coverage). Relative to the legacy combined report:

- 130 audited explicit verdicts move from abstain to committed;
- 4,526 unaudited semantic commits move to abstain;
- 74 audited non-semantic legacy AITA commits move to abstain;
- 18 legacy semantic rows are retained through an explicit digit appraisal;
- three legacy semantic rows are retained through an explicit AITA verdict; and
- no `semantic_unaudited` confidence remains.

All 130 new commits and all 74 non-semantic demotions were reviewed using the
complete response text. Across the immutable audit files, 262 unique affected
keys were inspected. A second run through the full parser pipeline agreed on
all 204 audited outcome changes (0 disagreements); manual-audit validation also
has 0 disagreements. The regression suite contains 67 cases and passes in full.
See the refinement `manifest.json`, `manual_adjudication.json`,
`manual_audit_validation.json`, and `full_pipeline_validation.json` for hashes
and machine-readable evidence.
