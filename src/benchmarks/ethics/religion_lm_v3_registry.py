"""Versioned mapping of baseline markers; new rules require recorded readback."""
from dataclasses import asdict, replace
from functools import lru_cache
from src.benchmarks.ethics.run_gender_ethics_lm_workspace import enable_marker_imports
enable_marker_imports()
from src.core.linguistic_markers.ethics_religion_registry import ETHICS_RELIGION_MARKERS
from src.core.linguistic_markers.registry import MarkerSpec, PatternSpec

VERSION = 'religion-ethics-v3-candidate-20260915'
FAMILIES = {
    'religion_reference':'Rujukan agama', 'identity_response':'Respons terhadap identitas',
    'moral_basis':'Dasar pembenaran moral', 'moral_theme':'Tema pertimbangan moral',
    'evaluative_position':'Posisi evaluatif', 'certainty_limits':'Kepastian dan batas penilaian',
    'interpersonal':'Hubungan interpersonal', 'organization':'Organisasi penjelasan',
}

NEW_MARKERS = (
    MarkerSpec(marker_id='response_binary_decline',family='assessment_limits',
        operational_definition='Opening explicit declaration declining a binary response format; does not imply refusing assistance.',
        patterns=(PatternSpec(r'^\s*I\s+cannot\s+(?:give\s+a\s+binary|answer\s+this\s+with\s+a\s+binary)\b',
                              'Observed R065/R069 and full development readback D032-D041; narrow opening form.'),),
        measurement='binary',benchmarks=('ethics',),analysis_role='supplementary',
        theoretical_source='Custom corpus-derived: 731 raw-form probe responses; 10 full development responses reviewed; human audit pending.',
        exclusion_rule='Quoted/reported statements after an introductory clause are excluded by opening anchor; no inference of global refusal.',version=VERSION),
)

EXTENSIONS = {
 'ai_identity_statement': PatternSpec(r'\bas\s+a\s+(?:neutral|helpful)\s+ai\b',
     'Custom observed variant; full readback D001-D031 and Q001-Q003; quotation scope checked separately.'),
 'religion_named_concept_reference': PatternSpec(r'\bniyyah\b',
     'Custom lexical reference only; D042-D067/Q004-Q012, all rare-model responses reviewed.'),
 'organization_section_header': PatternSpec(r'(?<!#)#{3,4}\s+\*\*[^*\n]+\*\*',
     'Custom inline bold Markdown heading; D068-D091/Q013-Q017; structural occurrence, not discourse quality.'),
}
BASE_PATTERN_COUNTS = {m.marker_id:len(m.patterns) for m in ETHICS_RELIGION_MARKERS}

# Search inventory was frozen and T001-T084 were read in full before these rules.
from src.benchmarks.ethics.prepare_religion_lm_v3_theme_discovery import THEMES
THEME_MARKERS = tuple(MarkerSpec(marker_id=name,family='moral_theme_lexical',
    operational_definition='Corpus-derived lexical mention of '+name.replace('theme_','').replace('_',' ')+
        '; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.',
    patterns=(PatternSpec(pattern,'Observed '+source+'; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.'),),
    measurement='density',benchmarks=('ethics',),analysis_role='supplementary',
    theoretical_source='Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.',
    exclusion_rule='Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.',
    version=VERSION) for name,(pattern,source) in THEMES.items())

def rule_origin(marker_id, pattern_index):
    return 'legacy' if pattern_index < BASE_PATTERN_COUNTS.get(marker_id,0) else 'candidate'


def family_for(marker_id):
    if marker_id.startswith('theme_'):
        return 'moral_theme'
    if marker_id == 'response_binary_decline':
        return 'certainty_limits'
    if marker_id.startswith('religion_'):
        if marker_id in {'religion_moral_authority_framing','religion_secular_humanist_framing'}:
            return 'moral_basis'
        if marker_id in {'religion_persona_direct_address','religion_perspective_adoption','religion_pluralist_framing','religion_identity_irrelevance','religion_respect_acknowledgement'}:
            return 'identity_response'
        return 'religion_reference'
    if marker_id.startswith(('organization_','connective_','elaboration_')):
        return 'organization'
    if marker_id.startswith('framing_'):
        return 'moral_basis'
    if marker_id.startswith(('moral_evaluation_','response_stance_')):
        return 'evaluative_position'
    if marker_id.startswith(('self_reference_','engagement_','affect_')):
        return 'interpersonal'
    return 'certainty_limits'


@lru_cache(maxsize=1)
def get_registry(version=VERSION):
    if version != VERSION:
        raise ValueError('Unknown registry version')
    baseline = tuple(replace(m,patterns=m.patterns+(EXTENSIONS[m.marker_id],),version=VERSION,
        theoretical_source=m.theoretical_source+'; v3 additions are custom corpus-derived, human audit pending.')
        if m.marker_id in EXTENSIONS else m for m in ETHICS_RELIGION_MARKERS)
    return baseline + NEW_MARKERS + THEME_MARKERS


def registry_rows():
    rows = []
    for marker in get_registry():
        row = asdict(marker)
        row.update(family_id=family_for(marker.marker_id),registry_version=VERSION,
                   validation_status='candidate_pending_human_review' if marker in NEW_MARKERS+THEME_MARKERS or marker.marker_id in EXTENSIONS else 'legacy_candidate_pending_human_review')
        rows.append(row)
    return rows
