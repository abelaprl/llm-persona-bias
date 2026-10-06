# Laporan linguistic marker ETHICS — candidate

Sumber memuat **216 respons**, 6 model, 9 persona, 4 subset, dan 41 marker. Default visual adalah overall seluruh subset; setiap model tetap terpisah.

Target recall ≥99% per marker belum terbukti. Semua grafik berstatus candidate; p-value dan CI tidak memvalidasi detektor. Audit manusia dan validasi parsing tetap menjadi gate analisis final.

## Denominator dan satuan

Seluruh respons sumber masuk coverage, termasuk missing/verdict-only. Profil marker memakai n_available setiap measurement; NA tidak diubah menjadi nol. Count adalah occurrence per respons tersedia, density per 100 kata, prevalence dalam persen; selisih presence dalam poin persentase. Raw occurrence dan denominator tersedia di marker_summary.csv.

| Status teks | Respons |
|---|---:|
| substantive_explanation | 172 |
| verdict_only | 43 |
| missing | 1 |

## Interpretasi perbandingan

Arah direct contrast selalu Woman−Man. Positif berarti nilai marker lebih tinggi pada woman. Common-triplet map menggunakan triplet utuh; neutral tidak dihitung sebagai empat observasi independen. CI adalah CI nominal 95%, bukan CI simultan Holm. n_changed <10 ditandai low_support, tanpa menghapus titik. Ukuran efek rank dan selisih rerata mempunyai estimand berbeda dan tidak saling dikurangkan.

Perbedaan persona implisit adalah efek kondisi prompt beserta narasinya; bukan isolasi gender murni. Kedekatan dengan diagonal atau p>0,05 bukan bukti ekuivalensi. Tidak ada filter significant-only atau committed-only.

## Akses bukti

Buka [laporan visual](report.html) atau [penjelajah respons penuh](evidence_explorer.html). figure_index.csv merekam source CSV, satuan, denominator, scope, status, dan SHA256 setiap figur. marker_summary.csv tetap memuat seluruh subset dan ketiga satuan. Default render adalah overall; render_subset_figures=True menambah atlas subset. Default forest/profile/triplet memakai measurement registry (binary → presence); full_detail=True merender ketiga measurement. Family registry tetap utuh saat keluarga kecil berbagi halaman dengan maksimum delapan marker.

## Indeks figur

| Jenis | Scope | Satuan | PNG | Data |
|---|---|---|---|---|
| coverage | overall |  | [PNG](figures/overall_status_3bf34505.png) | [CSV](figures/overall_status_3bf34505.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_reasoning_organization_1_count_f857f1b6.png) | [CSV](figures/atlas_overall_reasoning_organization_1_count_f857f1b6.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_reasoning_organization_1_density_155640e2.png) | [CSV](figures/atlas_overall_reasoning_organization_1_density_155640e2.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_reasoning_organization_1_presence_ebf20822.png) | [CSV](figures/atlas_overall_reasoning_organization_1_presence_ebf20822.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_epistemic_stance_2_count_060b057b.png) | [CSV](figures/atlas_overall_epistemic_stance_2_count_060b057b.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_epistemic_stance_2_density_af561c69.png) | [CSV](figures/atlas_overall_epistemic_stance_2_density_af561c69.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_epistemic_stance_2_presence_261d4baf.png) | [CSV](figures/atlas_overall_epistemic_stance_2_presence_261d4baf.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_logical_connective_self_reference_engagement_3_coun_ef3e56f5.png) | [CSV](figures/atlas_overall_logical_connective_self_reference_engagement_3_coun_ef3e56f5.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_logical_connective_self_reference_engagement_3_dens_47f7e9bb.png) | [CSV](figures/atlas_overall_logical_connective_self_reference_engagement_3_dens_47f7e9bb.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_logical_connective_self_reference_engagement_3_pres_6243c99e.png) | [CSV](figures/atlas_overall_logical_connective_self_reference_engagement_3_pres_6243c99e.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_textual_elaboration_model_identity_response_stance__a26e838c.png) | [CSV](figures/atlas_overall_textual_elaboration_model_identity_response_stance__a26e838c.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_textual_elaboration_model_identity_response_stance__363d6871.png) | [CSV](figures/atlas_overall_textual_elaboration_model_identity_response_stance__363d6871.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_textual_elaboration_model_identity_response_stance__a5d567c9.png) | [CSV](figures/atlas_overall_textual_elaboration_model_identity_response_stance__a5d567c9.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_b5e65d47.png) | [CSV](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_b5e65d47.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_82069516.png) | [CSV](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_82069516.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_43ec72ce.png) | [CSV](figures/atlas_overall_ethical_content_framing_moral_evaluation_language_5_43ec72ce.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_750f49a2.png) | [CSV](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_750f49a2.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_165a3401.png) | [CSV](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_165a3401.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_f71d107f.png) | [CSV](figures/atlas_overall_gender_persona_uptake_gender_conditional_identity_r_f71d107f.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_525c1102.png) | [CSV](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_525c1102.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_c91900ec.png) | [CSV](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_c91900ec.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_061f7d3b.png) | [CSV](figures/atlas_overall_gender_identity_discourse_gender_role_discourse_gen_061f7d3b.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_reasoning_organization_1_density_0_e5c0cbf5.png) | [CSV](figures/forest_neutral_reasoning_organization_1_density_0_e5c0cbf5.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_epistemic_stance_2_density_0_e2b0caf8.png) | [CSV](figures/forest_neutral_epistemic_stance_2_density_0_e2b0caf8.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_logical_connective_self_reference_engagement_3_den_146e8392.png) | [CSV](figures/forest_neutral_logical_connective_self_reference_engagement_3_den_146e8392.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_textual_elaboration_model_identity_response_stance_6ef5534d.png) | [CSV](figures/forest_neutral_textual_elaboration_model_identity_response_stance_6ef5534d.csv) |
| forest_neutral | overall | presence | [PNG](figures/forest_neutral_textual_elaboration_model_identity_response_stance_fe1cea80.png) | [CSV](figures/forest_neutral_textual_elaboration_model_identity_response_stance_fe1cea80.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_ethical_content_framing_moral_evaluation_language__00510f27.png) | [CSV](figures/forest_neutral_ethical_content_framing_moral_evaluation_language__00510f27.csv) |
| forest_neutral | overall | presence | [PNG](figures/forest_neutral_gender_persona_uptake_gender_conditional_identity__6a5f2523.png) | [CSV](figures/forest_neutral_gender_persona_uptake_gender_conditional_identity__6a5f2523.csv) |
| forest_neutral | overall | density | [PNG](figures/forest_neutral_gender_identity_discourse_gender_role_discourse_ge_26c6f629.png) | [CSV](figures/forest_neutral_gender_identity_discourse_gender_role_discourse_ge_26c6f629.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_reasoning_organization_1_density_0_cda3877c.png) | [CSV](figures/forest_woman_man_reasoning_organization_1_density_0_cda3877c.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_epistemic_stance_2_density_0_0a0755dd.png) | [CSV](figures/forest_woman_man_epistemic_stance_2_density_0_0a0755dd.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_logical_connective_self_reference_engagement_3_d_96f1052c.png) | [CSV](figures/forest_woman_man_logical_connective_self_reference_engagement_3_d_96f1052c.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_textual_elaboration_model_identity_response_stan_c6ebee74.png) | [CSV](figures/forest_woman_man_textual_elaboration_model_identity_response_stan_c6ebee74.csv) |
| forest_woman_man | overall | presence | [PNG](figures/forest_woman_man_textual_elaboration_model_identity_response_stan_9bd9e15a.png) | [CSV](figures/forest_woman_man_textual_elaboration_model_identity_response_stan_9bd9e15a.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_ethical_content_framing_moral_evaluation_languag_1fd44e00.png) | [CSV](figures/forest_woman_man_ethical_content_framing_moral_evaluation_languag_1fd44e00.csv) |
| forest_woman_man | overall | presence | [PNG](figures/forest_woman_man_gender_persona_uptake_gender_conditional_identit_24bd25c3.png) | [CSV](figures/forest_woman_man_gender_persona_uptake_gender_conditional_identit_24bd25c3.csv) |
| forest_woman_man | overall | density | [PNG](figures/forest_woman_man_gender_identity_discourse_gender_role_discourse__395e4fa3.png) | [CSV](figures/forest_woman_man_gender_identity_discourse_gender_role_discourse__395e4fa3.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_reasoning_organization_1_density_0_63cf11e4.png) | [CSV](figures/forest_implicit_reasoning_organization_1_density_0_63cf11e4.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_reasoning_organization_1_density_8_bf00a8ba.png) | [CSV](figures/forest_implicit_reasoning_organization_1_density_8_bf00a8ba.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_epistemic_stance_2_density_0_acc7ce38.png) | [CSV](figures/forest_implicit_epistemic_stance_2_density_0_acc7ce38.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_epistemic_stance_2_density_8_cd6b4c50.png) | [CSV](figures/forest_implicit_epistemic_stance_2_density_8_cd6b4c50.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_logical_connective_self_reference_engagement_3_de_e7b53f32.png) | [CSV](figures/forest_implicit_logical_connective_self_reference_engagement_3_de_e7b53f32.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_logical_connective_self_reference_engagement_3_de_61276f13.png) | [CSV](figures/forest_implicit_logical_connective_self_reference_engagement_3_de_61276f13.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_3db82e8b.png) | [CSV](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_3db82e8b.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_9f701a97.png) | [CSV](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_9f701a97.csv) |
| forest_implicit | overall | presence | [PNG](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_c32d9fc6.png) | [CSV](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_c32d9fc6.csv) |
| forest_implicit | overall | presence | [PNG](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_393f5e99.png) | [CSV](figures/forest_implicit_textual_elaboration_model_identity_response_stanc_393f5e99.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_ethical_content_framing_moral_evaluation_language_80285daa.png) | [CSV](figures/forest_implicit_ethical_content_framing_moral_evaluation_language_80285daa.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_ethical_content_framing_moral_evaluation_language_8232b245.png) | [CSV](figures/forest_implicit_ethical_content_framing_moral_evaluation_language_8232b245.csv) |
| forest_implicit | overall | presence | [PNG](figures/forest_implicit_gender_persona_uptake_gender_conditional_identity_31ee0365.png) | [CSV](figures/forest_implicit_gender_persona_uptake_gender_conditional_identity_31ee0365.csv) |
| forest_implicit | overall | presence | [PNG](figures/forest_implicit_gender_persona_uptake_gender_conditional_identity_0004866d.png) | [CSV](figures/forest_implicit_gender_persona_uptake_gender_conditional_identity_0004866d.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_gender_identity_discourse_gender_role_discourse_g_56b506cd.png) | [CSV](figures/forest_implicit_gender_identity_discourse_gender_role_discourse_g_56b506cd.csv) |
| forest_implicit | overall | density | [PNG](figures/forest_implicit_gender_identity_discourse_gender_role_discourse_g_5d432e8e.png) | [CSV](figures/forest_implicit_gender_identity_discourse_gender_role_discourse_g_5d432e8e.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_reasoning_organization_1_density_0_579b3fec.png) | [CSV](figures/forest_interaction_reasoning_organization_1_density_0_579b3fec.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_epistemic_stance_2_density_0_fbc5068b.png) | [CSV](figures/forest_interaction_epistemic_stance_2_density_0_fbc5068b.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_logical_connective_self_reference_engagement_3_0dd44e4a.png) | [CSV](figures/forest_interaction_logical_connective_self_reference_engagement_3_0dd44e4a.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_textual_elaboration_model_identity_response_st_39c6b4e0.png) | [CSV](figures/forest_interaction_textual_elaboration_model_identity_response_st_39c6b4e0.csv) |
| forest_interaction | overall | presence | [PNG](figures/forest_interaction_textual_elaboration_model_identity_response_st_e25c012f.png) | [CSV](figures/forest_interaction_textual_elaboration_model_identity_response_st_e25c012f.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_ethical_content_framing_moral_evaluation_langu_0778d3c4.png) | [CSV](figures/forest_interaction_ethical_content_framing_moral_evaluation_langu_0778d3c4.csv) |
| forest_interaction | overall | presence | [PNG](figures/forest_interaction_gender_persona_uptake_gender_conditional_ident_7469ac32.png) | [CSV](figures/forest_interaction_gender_persona_uptake_gender_conditional_ident_7469ac32.csv) |
| forest_interaction | overall | density | [PNG](figures/forest_interaction_gender_identity_discourse_gender_role_discours_659ee41a.png) | [CSV](figures/forest_interaction_gender_identity_discourse_gender_role_discours_659ee41a.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_reasoning_organization_1_density_0_92e7d223.png) | [CSV](figures/conditions_reasoning_organization_1_density_0_92e7d223.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_reasoning_organization_1_density_3_9b364eb1.png) | [CSV](figures/conditions_reasoning_organization_1_density_3_9b364eb1.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_epistemic_stance_2_density_0_bb991034.png) | [CSV](figures/conditions_epistemic_stance_2_density_0_bb991034.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_epistemic_stance_2_density_3_31a80533.png) | [CSV](figures/conditions_epistemic_stance_2_density_3_31a80533.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_logical_connective_self_reference_engagement_3_density_f03b5c79.png) | [CSV](figures/conditions_logical_connective_self_reference_engagement_3_density_f03b5c79.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_logical_connective_self_reference_engagement_3_density_621c781a.png) | [CSV](figures/conditions_logical_connective_self_reference_engagement_3_density_621c781a.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_textual_elaboration_model_identity_response_stance_aff_594dec8c.png) | [CSV](figures/conditions_textual_elaboration_model_identity_response_stance_aff_594dec8c.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_textual_elaboration_model_identity_response_stance_aff_ad3914e4.png) | [CSV](figures/conditions_textual_elaboration_model_identity_response_stance_aff_ad3914e4.csv) |
| condition_profile | overall | presence | [PNG](figures/conditions_textual_elaboration_model_identity_response_stance_aff_185b66ae.png) | [CSV](figures/conditions_textual_elaboration_model_identity_response_stance_aff_185b66ae.csv) |
| condition_profile | overall | presence | [PNG](figures/conditions_textual_elaboration_model_identity_response_stance_aff_26b3d55b.png) | [CSV](figures/conditions_textual_elaboration_model_identity_response_stance_aff_26b3d55b.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_ethical_content_framing_moral_evaluation_language_5_de_691411db.png) | [CSV](figures/conditions_ethical_content_framing_moral_evaluation_language_5_de_691411db.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_ethical_content_framing_moral_evaluation_language_5_de_da304867.png) | [CSV](figures/conditions_ethical_content_framing_moral_evaluation_language_5_de_da304867.csv) |
| condition_profile | overall | presence | [PNG](figures/conditions_gender_persona_uptake_gender_conditional_identity_refe_3f63fed3.png) | [CSV](figures/conditions_gender_persona_uptake_gender_conditional_identity_refe_3f63fed3.csv) |
| condition_profile | overall | presence | [PNG](figures/conditions_gender_persona_uptake_gender_conditional_identity_refe_eae35845.png) | [CSV](figures/conditions_gender_persona_uptake_gender_conditional_identity_refe_eae35845.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_gender_identity_discourse_gender_role_discourse_gender_2ebb8d65.png) | [CSV](figures/conditions_gender_identity_discourse_gender_role_discourse_gender_2ebb8d65.csv) |
| condition_profile | overall | density | [PNG](figures/conditions_gender_identity_discourse_gender_role_discourse_gender_416fdac7.png) | [CSV](figures/conditions_gender_identity_discourse_gender_role_discourse_gender_416fdac7.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_reasoning_organization_1_density_5135a741.png) | [CSV](figures/triplet_reasoning_organization_1_density_5135a741.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_epistemic_stance_2_density_3560abbd.png) | [CSV](figures/triplet_epistemic_stance_2_density_3560abbd.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_logical_connective_self_reference_engagement_3_density_7899a221.png) | [CSV](figures/triplet_logical_connective_self_reference_engagement_3_density_7899a221.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_textual_elaboration_model_identity_response_stance_affect_f9f2a98d.png) | [CSV](figures/triplet_textual_elaboration_model_identity_response_stance_affect_f9f2a98d.csv) |
| neutral_triplet_map | overall | presence | [PNG](figures/triplet_textual_elaboration_model_identity_response_stance_affect_4dba2b9c.png) | [CSV](figures/triplet_textual_elaboration_model_identity_response_stance_affect_4dba2b9c.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_ethical_content_framing_moral_evaluation_language_5_densi_6ca33a8e.png) | [CSV](figures/triplet_ethical_content_framing_moral_evaluation_language_5_densi_6ca33a8e.csv) |
| neutral_triplet_map | overall | presence | [PNG](figures/triplet_gender_persona_uptake_gender_conditional_identity_referen_29819a4a.png) | [CSV](figures/triplet_gender_persona_uptake_gender_conditional_identity_referen_29819a4a.csv) |
| neutral_triplet_map | overall | density | [PNG](figures/triplet_gender_identity_discourse_gender_role_discourse_gendered__fe2c8551.png) | [CSV](figures/triplet_gender_identity_discourse_gender_role_discourse_gendered__fe2c8551.csv) |
| audit | overall |  | [PNG](figures/audit_dashboard_1c5fe0ef.png) | [CSV](figures/audit_dashboard_1c5fe0ef.csv) |
