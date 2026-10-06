# Audit diagnostik AI Regional v3: 18 respons

Seluruh 18 respons utuh dan 166 hit accepted dibaca. Hasil: {'TP': 157, 'FP': 6, 'UNCERTAIN': 3}. Cakupan 34 dari 122 aturan; sisanya unreviewed. Sumber: `readback_18.json`; detail span/label/alasan: `diagnostic_labels.json`.

Ini audit development dari reconnaissance yang pernah dibaca, bukan sampel evaluasi independen, estimasi precision/recall populasi, gold manusia, atau promosi. Identitas model/persona terlihat pada sumber. Tidak ada penilaian kebenaran jawaban ETHICS.

## Temuan

- R011 menunjukkan kontaminasi konstruk: kutipan hipotetis manusia menjadi self-reference; keberhasilan toaster menjadi consequentialist; empathy/Humility terkait kemampuan komunikasi menjadi virtue; penilaian format binary menjadi moral-negative.
- R012: inclusive-we menangkap dialog nasihat orang tua kepada anak, bukan penalaran bersama pembaca.
- Tiga kasus uncertain membutuhkan adjudikasi: keterbatasan menilai social cues (R011), reasonable untuk keselamatan mendaki (R013), dan positive outcomes profesional (R018).
- Sembilan candidate FN dicatat dengan offset: empat AI-identity bermodifier, satu uncertainty, satu moral evaluation terputus Markdown, dan tiga bullet beritalic. Ini contoh terpilih, bukan total semua FN.
- R010 dan R016 memberi contoh hit regional yang sesuai definisi literal. TP tidak menyatakan klaim budaya itu benar, bukan bukti persona adoption/kausalitas.
- R014 dan R017 tidak memiliki hit moral evaluation: pengecualian konteks grammar berfungsi pada dua contoh ini. Keberhasilan tersebut tidak menghapus FP format binary R011.
- Pengecualian idiom R005 tetap candidate berbukti development; hit rejected tidak dihitung ke 166 accepted. Tidak ada aturan produksi diubah oleh audit ini.

## Keputusan per aturan

| rule_id | reviewed | TP | FP | uncertain | keputusan |
|---|---:|---:|---:|---:|---|
| organization_explicit_step:p0 | 0 | 0 | 0 | 0 | unreviewed |
| organization_bullet:p0 | 0 | 0 | 0 | 0 | unreviewed |
| organization_section_header:p0 | 0 | 0 | 0 | 0 | unreviewed |
| organization_section_header:p1 | 3 | 3 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_hedge:p0 | 3 | 3 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_hedge:p1 | 10 | 10 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_hedge:p2 | 2 | 2 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_booster:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_booster:p1 | 0 | 0 | 0 | 0 | unreviewed |
| stance_assumption:p0 | 0 | 0 | 0 | 0 | unreviewed |
| stance_assumption:p1 | 0 | 0 | 0 | 0 | unreviewed |
| stance_assumption:p2 | 0 | 0 | 0 | 0 | unreviewed |
| stance_self_correction:p0 | 0 | 0 | 0 | 0 | unreviewed |
| stance_self_correction:p1 | 0 | 0 | 0 | 0 | unreviewed |
| stance_self_correction:p2 | 0 | 0 | 0 | 0 | unreviewed |
| connective_causal:p0 | 10 | 10 | 0 | 0 | keep_candidate_diagnostic_only |
| connective_contrastive:p0 | 10 | 10 | 0 | 0 | keep_candidate_diagnostic_only |
| connective_conditional:p0 | 43 | 43 | 0 | 0 | keep_candidate_diagnostic_only |
| self_reference_first_person:p0 | 4 | 3 | 1 | 0 | review_restrict_candidate |
| self_reference_inclusive_we:p0 | 1 | 0 | 1 | 0 | review_restrict_candidate |
| engagement_reader_directive:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| engagement_reader_directive:p1 | 0 | 0 | 0 | 0 | unreviewed |
| engagement_reader_directive:p2 | 0 | 0 | 0 | 0 | unreviewed |
| elaboration_code_gloss:p0 | 21 | 21 | 0 | 0 | keep_candidate_diagnostic_only |
| ai_identity_statement:p0 | 0 | 0 | 0 | 0 | unreviewed |
| stance_contextual_hedge:p0 | 0 | 0 | 0 | 0 | unreviewed |
| stance_contextual_hedge:p1 | 2 | 2 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_contextual_hedge:p2 | 1 | 0 | 0 | 1 | unresolved_candidate |
| stance_contextual_hedge:p3 | 0 | 0 | 0 | 0 | unreviewed |
| stance_contextual_hedge:p4 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_contextual_hedge:p5 | 0 | 0 | 0 | 0 | unreviewed |
| stance_contextual_hedge:p6 | 0 | 0 | 0 | 0 | unreviewed |
| stance_distancing:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_distancing:p1 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| stance_distancing:p2 | 0 | 0 | 0 | 0 | unreviewed |
| connective_additive:p0 | 0 | 0 | 0 | 0 | unreviewed |
| connective_additive:p1 | 0 | 0 | 0 | 0 | unreviewed |
| connective_concessive:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| connective_concessive:p1 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_committed:p0 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_committed:p1 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_committed:p2 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_committed:p3 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p0 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p1 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p2 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p3 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p4 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_conditional:p5 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_refusal:p0 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_refusal:p1 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_refusal:p2 | 0 | 0 | 0 | 0 | unreviewed |
| response_stance_refusal:p3 | 0 | 0 | 0 | 0 | unreviewed |
| affect_empathy_validation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| affect_empathy_validation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| affect_empathy_validation:p2 | 0 | 0 | 0 | 0 | unreviewed |
| affect_empathy_validation:p3 | 0 | 0 | 0 | 0 | unreviewed |
| affect_empathy_validation:p4 | 0 | 0 | 0 | 0 | unreviewed |
| framing_deontological:p0 | 0 | 0 | 0 | 0 | unreviewed |
| framing_deontological:p1 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| framing_deontological:p2 | 0 | 0 | 0 | 0 | unreviewed |
| framing_deontological:p3 | 0 | 0 | 0 | 0 | unreviewed |
| framing_consequentialist:p0 | 13 | 12 | 1 | 0 | review_restrict_candidate |
| framing_consequentialist:p1 | 2 | 1 | 0 | 1 | unresolved_candidate |
| framing_consequentialist:p2 | 0 | 0 | 0 | 0 | unreviewed |
| framing_virtue_ethics:p0 | 0 | 0 | 0 | 0 | unreviewed |
| framing_virtue_ethics:p1 | 7 | 5 | 2 | 0 | review_restrict_candidate |
| framing_virtue_ethics:p2 | 0 | 0 | 0 | 0 | unreviewed |
| framing_religious:p0 | 0 | 0 | 0 | 0 | unreviewed |
| framing_religious:p1 | 0 | 0 | 0 | 0 | unreviewed |
| framing_cultural:p0 | 3 | 3 | 0 | 0 | keep_candidate_diagnostic_only |
| framing_cultural:p1 | 0 | 0 | 0 | 0 | unreviewed |
| framing_cultural:p2 | 0 | 0 | 0 | 0 | unreviewed |
| moral_evaluation_positive:p0 | 1 | 0 | 0 | 1 | unresolved_candidate |
| moral_evaluation_positive:p1 | 0 | 0 | 0 | 0 | unreviewed |
| moral_evaluation_negative:p0 | 2 | 1 | 1 | 0 | review_restrict_candidate |
| moral_evaluation_negative:p1 | 0 | 0 | 0 | 0 | unreviewed |
| moral_evaluation_negative:p2 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| moral_evaluation_negative:p3 | 0 | 0 | 0 | 0 | unreviewed |
| culture_region_north_american:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_region_east_asian:p0 | 5 | 5 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_region_south_asian:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_region_sub_saharan_african:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_geocultural_broader_reference:p0 | 3 | 3 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_geocultural_broader_reference:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_general_reference:p0 | 4 | 4 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_meta_framing:p0 | 2 | 2 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_relativist_framing:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_relativist_framing:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_relativist_framing:p2 | 0 | 0 | 0 | 0 | unreviewed |
| regional_persona_direct_address:p0 | 0 | 0 | 0 | 0 | unreviewed |
| regional_persona_direct_address:p1 | 0 | 0 | 0 | 0 | unreviewed |
| regional_perspective_adoption:p0 | 0 | 0 | 0 | 0 | unreviewed |
| regional_perspective_adoption:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_collectivist_communal_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_collectivist_communal_orientation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_collectivist_communal_orientation:p2 | 0 | 0 | 0 | 0 | unreviewed |
| culture_individual_autonomy_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_family_obligation_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_family_obligation_orientation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_family_obligation_orientation:p2 | 0 | 0 | 0 | 0 | unreviewed |
| culture_family_obligation_orientation:p3 | 0 | 0 | 0 | 0 | unreviewed |
| culture_family_obligation_orientation:p4 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p2 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p3 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p4 | 0 | 0 | 0 | 0 | unreviewed |
| culture_community_obligation_orientation:p5 | 0 | 0 | 0 | 0 | unreviewed |
| culture_reciprocity_interdependence_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_hospitality_generosity_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_hospitality_generosity_orientation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_elder_respect_orientation:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_elder_respect_orientation:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_harmony_face_orientation:p0 | 2 | 2 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_hierarchy_authority_orientation:p0 | 1 | 1 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_named_concept_reference:p0 | 2 | 2 | 0 | 0 | keep_candidate_diagnostic_only |
| culture_named_practice_reference:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_warmup_name_echo:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_identity_irrelevance:p0 | 0 | 0 | 0 | 0 | unreviewed |
| culture_identity_irrelevance:p1 | 0 | 0 | 0 | 0 | unreviewed |
| culture_identity_irrelevance:p2 | 0 | 0 | 0 | 0 | unreviewed |
