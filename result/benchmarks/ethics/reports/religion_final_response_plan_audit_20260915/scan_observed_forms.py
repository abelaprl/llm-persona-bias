"""Count observed discovery forms; these are search probes, not promoted rules."""
from pathlib import Path
import json
import re
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT / 'result/ETHICS/reports/Religion_Persona/_combined/all_models_results_full.csv'
MARKERS = ROOT / 'result/ETHICS/reports/Religion_Persona/linguistic_markers_religion_candidate_20260824_v1_1_10/response_level_markers.csv.gz'
# Each probe has a full-response discovery example and an existing comparison field.
PROBES = {
    'neutral_ai_identity': (r'\bas a neutral ai(?: assistant)?\b', 'ai_identity_statement', 'R013'),
    'helpful_ai_identity': (r'\bas a helpful ai assistant\b', 'ai_identity_statement', 'R014'),
    'explicit_binary_decline': (r'\bi cannot (?:give a binary|answer this with a binary)\b', 'response_stance_refusal', 'R065,R069'),
    'inline_markdown_heading': (r'(?<!\n)\s###\s+\*\*', 'organization_section_header', 'R037'),
    'niyyah': (r'\bniyyah\b', 'religion_named_concept_reference', 'R069'),
    'sabr': (r'\bsabr\b', 'religion_named_concept_reference', 'R037'),
    'neutrality_disclaimer': (r"\bi don['’]t have personal (?:opinions|beliefs|emotions)\b", 'ai_identity_statement', 'R003,R013,R021'),
}
paths = [OUT/'observed_forms_by_group.csv', OUT/'observed_forms_totals.csv', OUT/'observed_forms_manifest.json']
if any(p.exists() for p in paths):
    raise FileExistsError('Use a new output directory; existing evidence is immutable.')
keys = ['model_id', 'persona', 'question_id', 'subset']
marker_cols = sorted({f'{v[1]}__count' for v in PROBES.values()})
markers = pd.read_csv(MARKERS, usecols=keys+marker_cols)
groups = []
for chunk in pd.read_csv(SOURCE, usecols=['model','persona_folder','question_id','subset','response_content_full'], keep_default_na=False, dtype=str, chunksize=10000):
    chunk = chunk.rename(columns={'model':'model_id','persona_folder':'persona'})
    chunk = chunk.merge(markers, on=keys, validate='one_to_one')
    for probe, (pattern, marker, sample) in PROBES.items():
        hit = chunk.response_content_full.str.contains(pattern, flags=re.I, regex=True)
        matched = chunk.loc[hit, keys+[marker+'__count']].copy()
        matched['old_family_zero'] = matched[marker+'__count'].eq(0)
        group = matched.groupby(['model_id','persona','subset']).agg(probe_responses=('question_id','size'), old_family_zero=('old_family_zero','sum')).reset_index()
        group['probe'] = probe
        groups.append(group)
result = pd.concat(groups).groupby(['probe','model_id','persona','subset'], as_index=False)[['probe_responses','old_family_zero']].sum()
result.to_csv(paths[0],index=False)
totals = result.groupby('probe',as_index=False)[['probe_responses','old_family_zero']].sum()
totals.to_csv(paths[1],index=False)
paths[2].write_text(json.dumps({'probes':PROBES,'interpretation':'Observed-form search only. Old-family-zero is a review candidate, not an independently validated false negative. Nonzero family count does not prove this specific span was captured. Neutrality without AI mention is deliberately distinct from AI identity.','rows_scanned':154428,'ground_truth_read':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(totals.to_string(index=False))
