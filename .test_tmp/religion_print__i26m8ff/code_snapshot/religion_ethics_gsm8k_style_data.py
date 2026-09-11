"""Read frozen ETHICS v3 data for the GSM8K-style presentation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'result/ETHICS/reports/religion_lm_v3_candidate_20260916_r4_overall'
NOTEBOOK = ROOT / 'notebooks/religion/linguistic marker/ETHICS/linguistic_marker_religion_ethics_v3_candidate_20260916_r4_overall.ipynb'
IDENTITIES = ['islam', 'christian', 'jewish', 'atheist']
IDENTITY_LABELS = dict(zip(IDENTITIES, ['Islam', 'Christian', 'Jewish', 'Atheist']))
CONDITIONS = ['explicit', 'implicit_1', 'implicit_2', 'implicit_3']
CONDITION_LABELS = dict(zip(CONDITIONS, ['E', 'I1', 'I2', 'I3']))
MODELS = ['meta-llama_llama-3.1-8b-instruct', 'meta-llama_llama-3.3-70b-instruct',
          'mistralai_ministral-8b-2512', 'deepseek_deepseek-r1-distill-qwen-32b',
          'openai_gpt-oss-20b', 'qwen_qwen3-next-80b-a3b-thinking']
MODEL_LABELS = dict(zip(MODELS, ['Llama 3.1 8B', 'Llama 3.3 70B', 'Ministral 8B',
                               'DeepSeek R1 32B', 'GPT-OSS 20B', 'Qwen3 Next 80B']))
MODEL_COLORS = dict(zip(MODELS, ['#2563eb', '#7c3aed', '#059669', '#dc2626', '#d97706', '#0891b2']))
PERSONAS = ['neutral'] + [f'{i}_{c}' for i in IDENTITIES for c in CONDITIONS]
CONTRAST_KEY = ['model_id', 'contrast_kind', 'left_persona', 'right_persona', 'marker_id', 'metric']

LABELS = {
    'organization_explicit_step': 'Langkah bernomor', 'organization_bullet': 'Daftar bullet',
    'organization_section_header': 'Heading', 'stance_hedge': 'Hedge', 'stance_booster': 'Booster',
    'stance_assumption': 'Asumsi', 'stance_self_correction': 'Koreksi diri',
    'connective_causal': 'Konektif sebab', 'connective_contrastive': 'Konektif kontras',
    'connective_conditional': 'Konektif kondisional', 'self_reference_first_person': 'Rujukan diri',
    'self_reference_inclusive_we': 'We inklusif', 'engagement_reader_directive': 'Arahan pembaca',
    'elaboration_code_gloss': 'Penjelasan ulang', 'ai_identity_statement': 'Pernyataan identitas AI',
    'stance_contextual_hedge': 'Batas konteks', 'stance_distancing': 'Distancing',
    'connective_additive': 'Konektif tambahan', 'connective_concessive': 'Konektif konsesif',
    'response_stance_committed': 'Sikap tegas', 'response_stance_conditional': 'Sikap bersyarat',
    'response_stance_refusal': 'Refusal', 'affect_empathy_validation': 'Empati/validasi',
    'framing_deontological': 'Kewajiban/aturan', 'framing_consequentialist': 'Konsekuensi',
    'framing_virtue_ethics': 'Karakter/kebajikan', 'framing_religious': 'Framing agama',
    'framing_cultural': 'Framing budaya', 'moral_evaluation_positive': 'Evaluasi positif',
    'moral_evaluation_negative': 'Evaluasi negatif',
    'religion_identity_islam': 'Identitas Islam', 'religion_identity_christian': 'Identitas Christian',
    'religion_identity_jewish': 'Identitas Jewish', 'religion_identity_atheist': 'Identitas atheist',
    'religion_generic_reference': 'Rujukan agama umum', 'religion_deity_invocation': 'Tuhan/deity',
    'religion_figure_reference': 'Figur agama', 'religion_scripture_reference': 'Kitab agama',
    'religion_practice_community': 'Praktik/komunitas', 'religion_afterlife_soteriology': 'Akhirat/keselamatan',
    'religion_named_concept_reference': 'Konsep agama', 'religion_formulaic_expression': 'Sapaan/formula agama',
    'religion_persona_direct_address': 'Sapaan identitas pengguna',
    'religion_perspective_adoption': 'Pemakaian perspektif agama',
    'religion_moral_authority_framing': 'Otoritas moral agama',
    'religion_secular_humanist_framing': 'Framing secular/humanist',
    'religion_pluralist_framing': 'Framing pluralis', 'religion_identity_irrelevance': 'Identitas tidak relevan',
    'religion_blessing_invocation': 'Doa/blessing', 'religion_respect_acknowledgement': 'Pengakuan/respect',
    'response_binary_decline': 'Menolak format biner', 'theme_harm': 'Harm',
    'theme_compassion': 'Compassion', 'theme_fairness': 'Fairness',
    'theme_consent_autonomy': 'Consent/autonomy', 'theme_boundaries': 'Boundaries',
    'theme_responsibility': 'Responsibility', 'theme_intention': 'Intention', 'theme_honesty': 'Honesty/deception',
}
RELIGION_FOCUS = ['religion_generic_reference', 'religion_deity_invocation', 'religion_scripture_reference',
                  'religion_practice_community', 'religion_named_concept_reference', 'religion_formulaic_expression',
                  'religion_perspective_adoption', 'religion_moral_authority_framing',
                  'religion_identity_irrelevance', 'religion_blessing_invocation', 'religion_respect_acknowledgement']
GENERAL_FOCUS = ['stance_hedge', 'stance_booster', 'stance_contextual_hedge', 'stance_distancing',
                 'connective_causal', 'connective_contrastive', 'connective_conditional',
                 'self_reference_first_person', 'engagement_reader_directive', 'affect_empathy_validation',
                 'organization_bullet', 'organization_section_header', 'ai_identity_statement', 'response_binary_decline']
STANCE_FOCUS = ['response_stance_committed', 'response_stance_conditional', 'response_stance_refusal',
                'response_binary_decline', 'moral_evaluation_positive', 'moral_evaluation_negative']
MORAL_FOCUS = ['framing_deontological', 'framing_consequentialist', 'framing_virtue_ethics', 'framing_religious',
               'framing_cultural', 'religion_moral_authority_framing', 'religion_secular_humanist_framing']
THEME_FOCUS = [key for key in LABELS if key.startswith('theme_')]


def persona_parts(persona):
    if persona == 'neutral':
        return 'neutral', 'neutral'
    return tuple(persona.split('_', 1))


def add_persona_columns(frame, column='persona'):
    result = frame.copy()
    parts = result[column].map(persona_parts)
    result['identity'] = parts.map(lambda p: p[0])
    result['condition'] = parts.map(lambda p: p[1])
    return result


def coverage_table(rows):
    records = []
    for (model, persona), group in rows.groupby(['model_id', 'persona'], sort=False):
        text = group[group.surface_status.eq('linguistic_text')]
        words = text.word_count
        counts = group.surface_status.value_counts()
        status = group.answer_status.value_counts()
        records.append(dict(model_id=model, persona=persona, total=len(group),
                            linguistic_available=len(text), coverage_pct=100 * len(text) / len(group),
                            label_only=int(counts.get('label_only', 0)), empty_final=int(counts.get('empty_final', 0)),
                            technical_only=int(counts.get('technical_only', 0)),
                            committed=int(status.get('committed', 0)), abstain=int(status.get('abstain', 0)),
                            missing=int(status.get('missing', 0)), abstain_text=int(text.answer_status.eq('abstain').sum()),
                            word_total=int(words.sum()), median_words=words.median(),
                            q25_words=words.quantile(.25), q75_words=words.quantile(.75), mean_words=words.mean()))
    result = pd.DataFrame(records)
    for status in ['committed', 'abstain', 'missing']:
        result[('commit' if status == 'committed' else status) + '_rate'] = result[status] / result.total
    return add_persona_columns(result)


def select_contrasts(contrasts, kind, markers, metric='prevalence_text', population='text_all'):
    selected = contrasts.loc[contrasts.contrast_kind.eq(kind) & contrasts.population.eq(population)
                             & contrasts.metric.eq(metric) & contrasts.marker_id.isin(markers)].copy()
    if selected.duplicated(CONTRAST_KEY).any():
        raise ValueError('Duplicate source contrast keys')
    return add_persona_columns(selected, 'left_persona')


def sensitivity_table(contrasts):
    base = contrasts.loc[contrasts.population.eq('text_all'), CONTRAST_KEY + ['effect', 'n_pairs']]
    other = contrasts.loc[contrasts.population.isin(['both_committed_text', 'equal_subset_text', 'no_quality_flag_text']),
                          CONTRAST_KEY + ['population', 'effect', 'n_pairs']]
    result = other.merge(base, on=CONTRAST_KEY, how='left', validate='many_to_one', suffixes=('_sensitivity', '_main'))
    result['evaluable'] = np.isfinite(result.effect_main) & np.isfinite(result.effect_sensitivity)
    result['direction_reversed'] = result.evaluable & (result.effect_main * result.effect_sensitivity < 0)
    return result


def term_table(hits, rows):
    text = rows.loc[rows.surface_status.eq('linguistic_text') & rows.word_count.gt(0),
                    ['response_key', 'model_id', 'persona', 'word_count']]
    if text.response_key.duplicated().any():
        raise ValueError('Duplicate response keys')
    hits = hits.copy()
    hits['term'] = hits.matched_text.str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    hits = hits.drop_duplicates(['response_key', 'raw_start', 'raw_end', 'term'])
    counts = hits.groupby(['response_key', 'term']).size().rename('raw_count').reset_index()
    counts = counts.merge(text, on='response_key', validate='many_to_one')
    counts['density'] = counts.raw_count * 100 / counts.word_count
    sums = counts.groupby(['model_id', 'persona', 'term']).agg(raw_count=('raw_count', 'sum'),
                                                            density_sum=('density', 'sum'),
                                                            n_present=('response_key', 'size')).reset_index()
    denominator = text.groupby(['model_id', 'persona']).agg(n_text=('response_key', 'size'),
                                                          word_total=('word_count', 'sum')).reset_index()
    # Include zero-hit models for every observed term within an identity.
    vocabulary = sums[['persona', 'term']].drop_duplicates()
    result = denominator.merge(vocabulary, on='persona').merge(sums, on=['model_id', 'persona', 'term'], how='left')
    result[['raw_count', 'density_sum', 'n_present']] = result[['raw_count', 'density_sum', 'n_present']].fillna(0)
    result['mean_response_density'] = result.density_sum / result.n_text
    result['prevalence_pct'] = 100 * result.n_present / result.n_text
    return add_persona_columns(result)


def select_examples(packages, models=MODELS):
    selected = {}
    for filename, samples in packages.items():
        for index, sample in enumerate(samples):
            model = sample['model_id']
            if model in models and model not in selected:
                selected[model] = dict(sample, example_source_file=filename, example_source_index=index)
    missing = set(models) - set(selected)
    if missing:
        raise ValueError(f'Example packages do not cover models: {sorted(missing)}')
    return [selected[model] for model in models]


def load_data(source=SOURCE):
    source = Path(source)
    print('[1/5] Membaca hasil v3 beku', flush=True)
    columns = ['model_id', 'persona', 'subset', 'question_id', 'response_key', 'surface_status',
               'answer_status', 'word_count', 'raw_final_sha256']
    rows = pd.read_csv(source / 'response_level/markers.csv.gz', usecols=columns)
    if rows.response_key.duplicated().any():
        raise ValueError('Response keys are not unique')
    summary = add_persona_columns(pd.read_csv(source / 'statistics/overall_descriptive.csv'))
    contrasts = pd.read_csv(source / 'statistics/paired_contrasts.csv.gz')
    if contrasts.duplicated(CONTRAST_KEY + ['population']).any():
        raise ValueError('Contrast source contains duplicate keys')
    availability = add_persona_columns(pd.read_csv(source / 'statistics/paired_availability.csv'))
    registry = json.loads((source / 'registry/markers.json').read_text(encoding='utf-8'))
    print('[2/5] Mengagregasi span istilah tersimpan; tanpa aturan marker baru', flush=True)
    chunks = []
    columns = ['response_key', 'model_id', 'persona', 'raw_start', 'raw_end', 'matched_text', 'marker_id', 'family_id']
    processed = 0
    for chunk in pd.read_csv(source / 'occurrences/hits.csv.gz', usecols=columns, chunksize=200000):
        keep = chunk.family_id.eq('religion_reference') & chunk.persona.str.endswith('_explicit')
        chunks.append(chunk.loc[keep])
        processed += len(chunk)
        print(f'  Occurrence dibaca: {processed:,}', flush=True)
    terms = term_table(pd.concat(chunks, ignore_index=True), rows)
    return dict(rows=rows, summary=summary, contrasts=contrasts, availability=availability,
                registry=registry, coverage=coverage_table(rows), terms=terms,
                sensitivity=sensitivity_table(contrasts))
