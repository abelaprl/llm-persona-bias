# Regional + Neutral priority-candidate audit (2026-08-21)

## Scope and method

- Reviewed all 267 priority candidates one by one: 123 `aita_language` and 144 `direct_moral_phrase`.
- Read each full response and matched the direct-moral candidates against the source JSON `scenario`.
- Did not consult ground-truth labels.
- Did not change the parser, scoring, API outputs, or final combined CSV.
- The adjudicated CSV includes the source scenario, source JSON path, full response, verdict, expected stance, and audit reason.

## Results

| Candidate group | Reviewed | Safe commit: acceptable | Safe commit: wrong | Keep abstain |
|---|---:|---:|---:|---:|
| AITA language | 123 | 34 | 7 | 82 |
| Direct moral phrase | 144 | 36 | 24 | 84 |
| **Total** | **267** | **70** | **31** | **166** |

The 101 safe commits are manual adjudication outcomes. They are an upper-bound projection, not evidence that one broad deterministic rule can recover all 101 safely.

## Why 166 candidates must remain abstain

| Reason | AITA | Direct | Total |
|---|---:|---:|---:|
| Conditional or depends on unstated facts | 33 | 21 | 54 |
| NTA/YTA acronym conflicts with the prose | 24 | 0 | 24 |
| Meta-evaluation of the `<0 or 1>` answer format, not the scenario | 0 | 25 | 25 |
| No asserted verdict | 0 | 17 | 17 |
| Nonbinary verdict | 7 | 7 | 14 |
| Nonbinary/graded score | 10 | 3 | 13 |
| Off-topic, prior-turn leakage, or garbled | 1 | 11 | 12 |
| Explicit refusal without a usable stance | 7 | 0 | 7 |
| **Total** | **82** | **84** | **166** |

## Important case families

1. **AITA acronym/prose conflict.** Some responses literally say variants of “you are the asshole (NTA)” or describe the actor as unreasonable and then attach `NTA`. A token-only AITA rule would silently invert these cases. Examples: audit rows 48, 53, 58, 73, 84, 90, 97, 105, and 110.

2. **Conditional and nonbinary AITA answers.** Responses give both NTA and YTA branches, 0.5/0.5 ratings, middle scores, or “partially in the wrong.” These are genuine abstentions, not parser misses. Examples: rows 42, 56, 59, 71, 85, 91, 98, and 112.

3. **Prior-turn/persona leakage.** A repeated response about “Kwame fixing his friends’ falling out” appears for unrelated target scenarios such as a snowman, potluck, freight-depot work, an MTurk job, a gas-station cashier, and a church offering. A stance can be clear yet still be unusable because it answers the wrong scenario. Examples: rows 158–163, 167–168, 170, 172, and 174.

4. **Meta-answer-format verdicts.** Many Mistral/Qwen responses conclude `Wrong`, but they are judging whether replying with the literal placeholder `<0 or 1>` is natural communication. They are not judging the target action. Examples: rows 204–221, 227, 229, 233, 236, 241, 243–244, 246, and 257.

5. **Target inversion in justice items.** In row 237, the response says rushing an hour-long survey in 30 seconds is wrong while agreeing that the worker deserves rejection. The target claim is therefore acceptable even though the response contains a prominent final `Wrong`. A last-verdict heuristic would reverse it.

6. **Clear recoverable cases.** There are 41 recoverable AITA candidates and 60 recoverable direct candidates. Examples include explicit final NTA/YTA verdicts with consistent prose (rows 33, 61, 67, 121–122), direct conclusions such as “right thing to do” or `Answer: Wrong` (rows 124–127, 137–138, 150, and 184), and three explicit virtue-label mismatches (rows 254, 256, and 258).

## Coverage projection

Current final CSV:

- committed: 107,745
- abstain: 44,837
- missing: 1,846
- total: 154,428
- coverage over all rows: 69.770378%
- coverage over nonmissing rows: 70.614489%

If all 101 manually safe outcomes can be implemented with exact precision:

- committed: 107,846
- abstain: 44,736
- missing: 1,846 (unchanged)
- coverage over all rows: 69.835781% (**+0.065403 percentage points**)
- coverage over nonmissing rows: 70.680683% (**+0.066194 percentage points**)

## Parser recommendation before any change

Do not promote the broad AITA-token or direct-moral-phrase candidate rules. Safe implementation needs all of these safeguards:

1. Reject multiple, conditional, graded, or conflicting AITA verdicts before accepting an acronym.
2. Require agreement between the AITA acronym and nearby/full-response prose.
3. Reject meta-language that judges the answer format (`0/1`, “binary response,” “natural communication”) rather than the target scenario.
4. Make scenario alignment available to the parser or a validation layer; response-only rules cannot reliably identify prior-turn leakage.
5. Treat justice entitlement statements and virtue trait associations as target-specific structures, not generic occurrences of `right` or `wrong`.
6. Add every promoted case and every nearby counterexample to the regression suite before rebuilding the final CSV.

## Provenance

- Queue SHA-256: `162930481e0eedf46bd4a8a2686461ecf6408935f4bd2aa29b7dda4619474c84`
- Adjudicated CSV SHA-256: `c06737ed14b09b1105b5e776c3875f2f4cfe5d9354e3b9a364f95f968a124f37`
- Unchanged final CSV SHA-256: `ee3dc2e8f4242a106b0265b7fbfeeb0e1028e592fad7f0bee5a777cddbfb164d`

