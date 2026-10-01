"""Exhaustive candidate-construct discovery for ETHICS Regional/Cultural.

This is a recall-screening supplement, not an operational marker extractor.
Every visible response in the frozen v1.0.16 review queue is scanned against a
deliberately broader inventory of cultural-theory and discourse formulations.
Hits are written to a new, write-once audit directory and are never promoted to
the 23-marker registry by this script.  Ground truth and hidden reasoning are
not read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.csv as pacsv


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = (
    REPO_ROOT / "result" / "ETHICS" / "reports" / "Regional_Cultural_Persona"
)
QUEUE_PATH = (
    REPORT_ROOT
    / "linguistic_markers_regional_cultural_candidate_20260822_v1_0_16_exhaustive_review_queue_v1"
    / "response_review_queue.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    REPORT_ROOT
    / "linguistic_markers"
    / "audit"
    / "regional_cultural_ethics_candidate_construct_discovery_v2_0_0_20260827"
)

EXPECTED_ROWS = 154_428
EXPECTED_QUEUE_SHA256 = (
    "d9015ec60761ac9efdf9ff15585112405fcb9f77f099e6642fc5654db5bcca9e"
)


@dataclass(frozen=True)
class CandidateConstruct:
    candidate_id: str
    candidate_family: str
    operational_screen: str
    pattern: str
    related_registered_markers: str
    suggested_source: str
    source_status: str
    promotion_status: str = "candidate_not_promoted"


_CULTURAL_HEAD = (
    r"(?:cultures?|cultural\s+(?:contexts?|values?|norms?|traditions?|"
    r"perspectives?|practices?|identit(?:y|ies))|values?|norms?|traditions?|"
    r"communities|societies|perspectives?|contexts?|worldviews?)"
)

_EXTENDED_GEOCULTURAL = (
    r"(?:canadian|mexican|united\s+states|u\.?s\.?|taiwanese|mongolian|"
    r"hong\s+kong(?:er)?|macanese|afghan|bhutanese|maldivian|nepali|nepalese|"
    r"sri[- ]lankan|angolan|beninese|botswanan|burundian|cameroonian|chadian|"
    r"comorian|congolese|ivorian|djiboutian|eritrean|ethiopian|gabonese|"
    r"gambian|guinean|liberian|malagasy|malawian|malian|mauritian|mozambican|"
    r"namibian|nigerien|rwandan|senegalese|seychellois|somali|south[- ]sudanese|"
    r"sudanese|tanzanian|togolese|ugandan|zambian|zimbabwean|southeast[- ]asian|"
    r"central[- ]asian|middle[- ]eastern|latin[- ]american|european|oceanian)"
)


CANDIDATES: tuple[CandidateConstruct, ...] = (
    CandidateConstruct(
        "culture_tightness_looseness",
        "theoretical_dimension",
        "Explicit cultural tightness/looseness terminology.",
        r"(?i)\b(?:cultural\s+(?:tightness|looseness)|tightness[- ]looseness|"
        r"tight\s+cultures?|loose\s+cultures?)\b",
        "",
        "Gelfand et al. (2011), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_uncertainty_avoidance",
        "theoretical_dimension",
        "Explicit uncertainty-avoidance terminology.",
        r"(?i)\b(?:uncertainty\s+avoidance|avoidance\s+of\s+uncertainty)\b",
        "culture_hierarchy_authority_orientation",
        "Hofstede (2001), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_temporal_orientation",
        "theoretical_dimension",
        "Explicit long-/short-term cultural orientation terminology.",
        r"(?i)\b(?:long[- ]term|short[- ]term)\s+(?:cultural\s+)?orientation\b",
        "",
        "Hofstede (2001), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_honor_shame_dignity_system",
        "theoretical_dimension",
        "Names an honor, shame, dignity, or face culture/system.",
        r"(?i)\b(?:(?:honou?r|shame|dignity|face)\s+cultures?|culture\s+of\s+"
        r"(?:honou?r|shame|dignity|face)|(?:honou?r|shame|dignity|face)[- ]based\s+"
        r"(?:culture|system|norms?))\b",
        "culture_harmony_face_orientation",
        "Leung and Cohen (2011), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_egalitarianism",
        "theoretical_dimension",
        "Explicit egalitarian cultural value or orientation.",
        r"(?i)\b(?:egalitarianism|egalitarian\s+(?:cultures?|values?|norms?|"
        r"societies|orientation))\b",
        "culture_hierarchy_authority_orientation",
        "Schwartz (2006), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_universalism_particularism",
        "theoretical_dimension",
        "Explicit universalism/particularism terminology.",
        r"(?i)\b(?:universalism|particularism|universalist|particularist)\b",
        "",
        "Trompenaars and Hampden-Turner (1998), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_high_low_context_communication",
        "communication_style",
        "Explicit high-/low-context culture or communication terminology.",
        r"(?i)\b(?:high[- ]context|low[- ]context)\s+(?:cultures?|communication|"
        r"communities|societies|style)\b",
        "",
        "Hall (1976), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_ingroup_orientation",
        "relational_orientation",
        "Explicit in-group loyalty, obligation, bias, or preference.",
        r"(?i)\b(?:in[- ]group|ingroup)\s+(?:loyalty|obligations?|bias|"
        r"preference|solidarity|favoritism|favouritism)\b",
        "culture_collectivist_communal_orientation",
        "Triandis (1995), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_self_construal",
        "relational_orientation",
        "Explicit relational/interdependent/independent self-construal.",
        r"(?i)\b(?:relational\s+self|(?:interdependent|independent)\s+"
        r"self(?:[- ]construal)?|self[- ]construal)\b",
        "culture_collectivist_communal_orientation;culture_individual_autonomy_orientation",
        "Markus and Kitayama (1991), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_monochronic_polychronic",
        "communication_style",
        "Explicit monochronic/polychronic time orientation.",
        r"(?i)\b(?:monochronic|polychronic)(?:\s+(?:time|culture|orientation))?\b",
        "",
        "Hall (1983), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_ancestor_reverence",
        "named_value_or_practice",
        "Explicit ancestor worship/reverence/respect framing.",
        r"(?i)\b(?:ancestor\s+worship|ancestral\s+(?:reverence|veneration)|"
        r"(?:reverence|veneration|respect)\s+for\s+(?:one['’]s\s+)?ancestors?)\b",
        "culture_elder_respect_orientation;culture_named_concept_reference",
        "Custom recall expansion; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_extended_kin_network",
        "relational_orientation",
        "Explicit extended-family or kinship-network framing.",
        r"(?i)\b(?:extended\s+famil(?:y|ies)|kinship\s+networks?|extended\s+"
        r"kin(?:ship)?\s+(?:network|ties))\b",
        "culture_family_obligation_orientation",
        "Custom recall expansion; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_acculturation_assimilation",
        "identity_process",
        "Explicit acculturation, assimilation, or cultural adaptation.",
        r"(?i)\b(?:acculturat(?:e|es|ed|ing|ion)|assimilati(?:on|ve)|"
        r"cultural\s+adaptation|adapt(?:s|ed|ing)?\s+to\s+(?:a\s+|the\s+)?"
        r"(?:local\s+)?culture)\b",
        "culture_general_reference;culture_meta_framing",
        "Berry (1997), suggested; not yet in thesis bibliography",
        "external_theory_recall_screen",
    ),
    CandidateConstruct(
        "culture_appropriation",
        "identity_process",
        "Explicit cultural appropriation terminology.",
        r"(?i)\b(?:cultural\s+appropriation|appropriat(?:e|es|ed|ing)\s+"
        r"(?:another|other)\s+culture['’]s)\b",
        "culture_general_reference",
        "Custom recall expansion; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_indigenous_traditional_knowledge",
        "knowledge_system",
        "Explicit indigenous/traditional knowledge system.",
        r"(?i)\b(?:indigenous\s+(?:knowledge|worldviews?|traditions?|practices?)|"
        r"traditional\s+(?:ecological\s+)?knowledge)\b",
        "culture_general_reference",
        "Custom recall expansion; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_custom_explicit",
        "lexical_gap",
        "Context-bounded custom/customs formulations excluded by v1.0.16.",
        r"(?i)\b(?:(?:regional|local|social|religious|dating|tipping|cultural|"
        r"community|traditional)\s+customs?|customs?\s+(?:var(?:y|ies)|"
        r"differ(?:s|ed)?|across\s+cultures?|between\s+cultures?))\b",
        "culture_general_reference;culture_relativist_framing",
        "Full-corpus residual review on 2026-08-27; source required before promotion",
        "corpus_observed_gap",
    ),
    CandidateConstruct(
        "culture_universal_override",
        "cultural_discourse",
        "States that an ethical norm applies regardless of/across cultures.",
        r"(?i)\b(?:regardless\s+of\s+(?:one['’]s\s+|your\s+|their\s+|the\s+)?"
        r"culture|irrespective\s+of\s+(?:cultural\s+)?(?:background|culture)|"
        r"across\s+all\s+cultures?|universal(?:ly)?\s+[^.!?\n]{0,35}\s+"
        r"across\s+cultures?)\b",
        "culture_identity_irrelevance;culture_general_reference",
        "Custom discourse marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_relativism_alternative_form",
        "cultural_discourse",
        "Culture is said to shape/influence an interpretation or judgment.",
        r"(?i)\b(?:cultures?|cultural\s+(?:background|context|norms?|values?))\b"
        r"[^.!?\n]{0,70}\b(?:shape|shapes|shaped|influence|influences|influenced|"
        r"affect|affects|affected|determine|determines)\b",
        "culture_relativist_framing;culture_meta_framing",
        "Custom recall expansion of registered relativist framing",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_stereotype_caution",
        "cultural_discourse",
        "Explicit warning against stereotyping or overgeneralizing cultures.",
        r"(?i)\b(?:avoid(?:ing)?\s+(?:cultural\s+)?stereotyp(?:e|es|ing)|"
        r"cultural\s+stereotyp(?:e|es|ing)|not\s+all\s+(?:members?\s+of\s+)?"
        r"(?:a\s+|the\s+)?culture|overgeneraliz(?:e|es|ed|ing|ation)\s+"
        r"(?:about\s+)?cultures?)\b",
        "culture_general_reference;culture_meta_framing",
        "Custom discourse marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_cross_inter_multicultural",
        "cultural_discourse",
        "Explicit cross-cultural, intercultural, or multicultural framing.",
        r"(?i)\b(?:cross[- ]cultural|intercultural|multicultural)\b",
        "culture_general_reference;culture_meta_framing",
        "Custom recall expansion; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_conflict_compatibility",
        "cultural_discourse",
        "Explicit culture clash or conflict between cultural values.",
        r"(?i)\b(?:culture\s+clash|clash\s+of\s+cultures?|cultural\s+conflict|"
        r"conflicting\s+cultural\s+(?:values?|norms?|expectations?))\b",
        "culture_general_reference;culture_meta_framing",
        "Custom discourse marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_bicultural_diaspora_identity",
        "identity_process",
        "Explicit bicultural, diaspora, or hybrid cultural identity.",
        r"(?i)\b(?:bicultural|diaspor(?:a|ic)|hybrid\s+cultural\s+identity|"
        r"multicultural\s+identity)\b",
        "culture_general_reference;culture_meta_framing",
        "Custom identity marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_humility_competence",
        "cultural_discourse",
        "Explicit cultural humility, competence, or competency.",
        r"(?i)\b(?:cultural\s+(?:humility|competence|competency)|culturally\s+"
        r"competent)\b",
        "culture_meta_framing",
        "Custom discourse marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_code_switching",
        "communication_style",
        "Explicit code-switching terminology.",
        r"(?i)\bcode[- ]switch(?:ing|es|ed)?\b",
        "",
        "Custom communication marker; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_extended_geocultural_reference",
        "expanded_lexicon",
        "Cultural-head construction with a demonym/region outside the frozen list.",
        rf"(?i)\b{_EXTENDED_GEOCULTURAL}\s+{_CULTURAL_HEAD}\b",
        "culture_geocultural_broader_reference",
        "Taxonomy recall expansion; no construct promotion implied",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_extended_named_concept",
        "expanded_lexicon",
        "Named cultural concepts absent from the frozen corpus-observed list.",
        r"(?i)\b(?:mianzi|amae|jeong|nunchi|ikigai|honne|tatemae|jugaad|"
        r"satyagraha|sarvodaya|izzat|lokasangraha|atithi\s+devo\s+bhava|"
        r"teranga|botho|hunhu|unhu|ukama|palaver)\b",
        "culture_named_concept_reference",
        "Open-vocabulary recall screen; source required before promotion",
        "corpus_recall_screen",
    ),
    CandidateConstruct(
        "culture_essentializing_attribution",
        "cultural_discourse",
        "A cultural group is generalized as tending to value/emphasize a trait.",
        rf"(?i)\b(?:{_EXTENDED_GEOCULTURAL}|north[- ]american|east[- ]asian|"
        r"south[- ]asian|sub[- ]saharan[- ]african|african|asian|western|chinese|"
        r"japanese|korean|indian|pakistani|bangladeshi|nigerian|ghanaian|kenyan|"
        r"south[- ]african)\s+(?:cultures?|communities|societies|people)\b"
        r"[^.!?\n]{0,75}\b(?:tend\s+to|typically|generally|often)\s+"
        r"(?:value|emphasize|prioritize|favour|favor)\b",
        "culture_geocultural_broader_reference",
        "Custom essentialization screen; source required before promotion",
        "corpus_recall_screen",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, default=QUEUE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _context(text: str, start: int, end: int) -> tuple[str, str]:
    left = max(0, start - 180)
    right = min(len(text), end + 180)
    raw = re.sub(r"\s+", " ", text[left:right]).strip()
    local_start = start - left
    local_end = end - left
    signature = text[left:start] + " <MATCH> " + text[end:right]
    signature = re.sub(r"\d+(?:[.,]\d+)?", "<NUM>", signature.casefold())
    signature = re.sub(r"\s+", " ", signature).strip()
    return raw, signature


def _load_queue(path: Path):
    include = [
        "response_key",
        "model_id",
        "persona",
        "question_id",
        "subset",
        "answer_status",
        "text_status",
        "present_regional_markers",
        "visible_response",
    ]
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(include_columns=include),
    )


def scan(table, candidates: Sequence[CandidateConstruct]):
    texts = pc.fill_null(table["visible_response"], "")
    metadata = table.drop(["visible_response"]).to_pandas()
    text_py = table["visible_response"].to_pylist()
    hit_response_rows: list[dict[str, object]] = []
    occurrence_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    for number, candidate in enumerate(candidates, start=1):
        counts_arrow = pc.count_substring_regex(texts, candidate.pattern)
        counts = np.asarray(counts_arrow.to_numpy(zero_copy_only=False), dtype=np.int64)
        hit_indices = np.flatnonzero(counts > 0)
        python_pattern = re.compile(candidate.pattern)
        candidate_occurrences = 0
        registered_overlap_responses = 0

        for index in hit_indices:
            text = text_py[int(index)] or ""
            matches = list(python_pattern.finditer(text))
            if len(matches) != int(counts[index]):
                raise ValueError(
                    f"Arrow/Python match-count mismatch for {candidate.candidate_id} "
                    f"at row {index}: arrow={counts[index]}, python={len(matches)}"
                )
            meta = metadata.iloc[int(index)].to_dict()
            present = str(meta.get("present_regional_markers") or "")
            has_registered = bool(present and present.casefold() != "nan")
            registered_overlap_responses += int(has_registered)
            hit_response_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    **meta,
                    "candidate_occurrences": len(matches),
                    "has_any_registered_regional_marker": has_registered,
                    "visible_response": text,
                }
            )
            for match in matches:
                raw_context, signature = _context(text, match.start(), match.end())
                occurrence_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        **meta,
                        "matched_form": match.group(0),
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "has_any_registered_regional_marker": has_registered,
                        "context": raw_context,
                        "context_signature": signature,
                    }
                )
            candidate_occurrences += len(matches)

        inventory_rows.append(
            {
                **asdict(candidate),
                "response_hits": len(hit_indices),
                "occurrence_hits": candidate_occurrences,
                "responses_with_any_registered_regional_marker": registered_overlap_responses,
                "responses_without_registered_regional_marker": (
                    len(hit_indices) - registered_overlap_responses
                ),
                "review_quota": (
                    "review_all" if len(hit_indices) < 50 else "review_minimum_100"
                ),
                "ai_review_status": "pending_full_context_review",
                "human_review_status": "pending_two_independent_reviewers",
            }
        )
        print(
            f"[scan {number:02d}/{len(candidates):02d}] {candidate.candidate_id}: "
            f"{len(hit_indices):,} responses / {candidate_occurrences:,} occurrences",
            flush=True,
        )

    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(hit_response_rows),
        pd.DataFrame(occurrence_rows),
    )


def build_clusters(occurrences: pd.DataFrame) -> pd.DataFrame:
    if occurrences.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "matched_form_normalized",
                "context_signature",
                "occurrences",
                "responses",
                "models",
                "personas",
                "subsets",
                "example_response_key",
                "example_context",
                "ai_disposition",
                "ai_note",
                "human_reviewer_1",
                "human_reviewer_2",
            ]
        )
    work = occurrences.copy()
    work["matched_form_normalized"] = (
        work["matched_form"].str.casefold().str.replace(r"\s+", " ", regex=True)
    )
    grouped = work.groupby(
        ["candidate_id", "matched_form_normalized", "context_signature"],
        observed=True,
        sort=False,
    )
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": keys[0],
                "matched_form_normalized": keys[1],
                "context_signature": keys[2],
                "occurrences": len(group),
                "responses": group["response_key"].nunique(),
                "models": group["model_id"].nunique(),
                "personas": group["persona"].nunique(),
                "subsets": group["subset"].nunique(),
                "example_response_key": first["response_key"],
                "example_context": first["context"],
                "ai_disposition": "pending",
                "ai_note": "",
                "human_reviewer_1": "pending",
                "human_reviewer_2": "pending",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_id", "occurrences", "matched_form_normalized"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def main() -> int:
    args = parse_args()
    queue_path = args.queue_path.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output is write-once and already exists: {output_dir}")
    if sha256_file(queue_path) != EXPECTED_QUEUE_SHA256:
        raise ValueError("Frozen review-queue SHA-256 mismatch.")

    output_dir.mkdir(parents=True, exist_ok=False)
    print("[1/4] Load frozen response review queue with Arrow", flush=True)
    table = _load_queue(queue_path)
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {table.num_rows:,}.")

    print("[2/4] Scan every response against candidate constructs", flush=True)
    inventory, hit_responses, occurrences = scan(table, CANDIDATES)
    print("[3/4] Build lossless response and context inventories", flush=True)
    clusters = build_clusters(occurrences)

    inventory.to_csv(output_dir / "candidate_construct_inventory.csv", index=False)
    hit_responses.to_csv(
        output_dir / "candidate_hit_responses_full_text.csv.gz",
        index=False,
        compression="gzip",
    )
    occurrences.to_csv(
        output_dir / "candidate_occurrence_contexts.csv.gz",
        index=False,
        compression="gzip",
    )
    clusters.to_csv(
        output_dir / "candidate_context_clusters_review_queue.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = {
        "audit_version": "regional-cultural-ethics-candidate-construct-discovery-v2.0.0",
        "status": "candidate_discovery_complete_ai_and_human_semantic_review_pending",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "queue_path": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "rows_scanned": table.num_rows,
        "candidate_constructs": len(CANDIDATES),
        "candidate_response_hits": len(hit_responses),
        "unique_responses_with_candidate_hit": (
            int(hit_responses["response_key"].nunique()) if not hit_responses.empty else 0
        ),
        "candidate_occurrences": len(occurrences),
        "context_clusters": len(clusters),
        "contracts": {
            "every_queue_row_scanned": table.num_rows == EXPECTED_ROWS,
            "ground_truth_used": False,
            "hidden_reasoning_used": False,
            "operational_registry_modified": False,
            "candidate_rules_promoted": False,
            "full_visible_response_preserved_for_every_hit": (
                len(hit_responses) == int(inventory["response_hits"].sum())
            ),
            "every_occurrence_has_context": (
                len(occurrences) == int(inventory["occurrence_hits"].sum())
            ),
            "human_review_complete": False,
        },
        "interpretation": {
            "candidate_hit_is_operational_marker": False,
            "zero_hit_proves_construct_absent": False,
            "semantic_recall_100_percent_claimed": False,
            "purpose": "bounded recall expansion without contaminating frozen v1.0.16",
        },
    }
    (output_dir / "candidate_discovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = """# Supplement audit kandidat konstrak Regional/Cultural v2.0.0

Semua 154.428 baris pada review queue beku dipindai. Folder ini **tidak**
mengubah atau mempromosikan 23 marker operasional v1.0.16. Satu hit hanya
berarti formulasi kandidat muncul dan harus dibaca dalam respons utuh.

- `candidate_construct_inventory.csv`: definisi, pola, sumber usulan, dan hit;
- `candidate_hit_responses_full_text.csv.gz`: setiap respons hit secara utuh;
- `candidate_occurrence_contexts.csv.gz`: setiap occurrence dan konteksnya;
- `candidate_context_clusters_review_queue.csv.gz`: antrean audit semantik;
- `candidate_discovery_summary.json`: kontrak dan provenance.

Promosi memerlukan audit presisi sesuai AGENTS.md dan dua reviewer manusia
independen. Kandidat dengan sumber di luar bibliografi tesis juga memerlukan
penambahan serta pemeriksaan referensi sebelum dapat disebut marker tesis.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    files = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**summary, "output_dir": str(output_dir), "generated_file_sha256": files}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[4/4] Wrote {len(CANDIDATES)} candidates, {len(hit_responses):,} "
        f"response hits, and {len(occurrences):,} occurrences to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
