# Laporan linguistic marker ETHICS — candidate

Sumber memuat **54 respons**, 6 model, 9 persona, 1 subset, dan 8 marker. Default visual adalah overall seluruh subset; setiap model tetap terpisah.

Target recall ≥99% per marker belum terbukti. Semua grafik berstatus candidate; p-value dan CI tidak memvalidasi detektor. Audit manusia dan validasi parsing tetap menjadi gate analisis final.

## Denominator dan satuan

Seluruh respons sumber masuk coverage, termasuk missing/verdict-only. Profil marker memakai n_available setiap measurement; NA tidak diubah menjadi nol. Count adalah occurrence per respons tersedia, density per 100 kata, prevalence dalam persen; selisih presence dalam poin persentase. Raw occurrence dan denominator tersedia di marker_summary.csv.

| Status teks | Respons |
|---|---:|
| substantive_explanation | 54 |

## Interpretasi perbandingan

Arah direct contrast selalu Woman−Man. Positif berarti nilai marker lebih tinggi pada woman. Common-triplet map menggunakan triplet utuh; neutral tidak dihitung sebagai empat observasi independen. CI adalah CI nominal 95%, bukan CI simultan Holm. n_changed <10 ditandai low_support, tanpa menghapus titik. Ukuran efek rank dan selisih rerata mempunyai estimand berbeda dan tidak saling dikurangkan.

Perbedaan persona implisit adalah efek kondisi prompt beserta narasinya; bukan isolasi gender murni. Kedekatan dengan diagonal atau p>0,05 bukan bukti ekuivalensi. Tidak ada filter significant-only atau committed-only.

## Akses bukti

Buka [laporan visual](report.html) atau [penjelajah respons penuh](evidence_explorer.html). figure_index.csv merekam source CSV, satuan, denominator, scope, status, dan SHA256 setiap figur. Filter subset pada laporan HTML membuka atlas supplementary.

Perbandingan berpasangan belum tersedia pada input ini; forest tidak dibuat.

Input common-triplet belum tersedia; peta relatif neutral tidak dibuat.

## Indeks figur

| Jenis | Scope | Satuan | PNG | Data |
|---|---|---|---|---|
| coverage | overall |  | [PNG](figures/overall_status_3bf34505.png) | [CSV](figures/overall_status_3bf34505.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_reasoning_organization_1_count_f857f1b6.png) | [CSV](figures/atlas_overall_reasoning_organization_1_count_f857f1b6.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_reasoning_organization_1_density_155640e2.png) | [CSV](figures/atlas_overall_reasoning_organization_1_density_155640e2.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_reasoning_organization_1_presence_ebf20822.png) | [CSV](figures/atlas_overall_reasoning_organization_1_presence_ebf20822.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_epistemic_stance_1_count_0baf57c8.png) | [CSV](figures/atlas_overall_epistemic_stance_1_count_0baf57c8.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_epistemic_stance_1_density_70eb4231.png) | [CSV](figures/atlas_overall_epistemic_stance_1_density_70eb4231.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_epistemic_stance_1_presence_74e61864.png) | [CSV](figures/atlas_overall_epistemic_stance_1_presence_74e61864.csv) |
| marker_atlas | overall | count | [PNG](figures/atlas_overall_logical_connective_1_count_9e1c47e5.png) | [CSV](figures/atlas_overall_logical_connective_1_count_9e1c47e5.csv) |
| marker_atlas | overall | density | [PNG](figures/atlas_overall_logical_connective_1_density_7ccaf0a0.png) | [CSV](figures/atlas_overall_logical_connective_1_density_7ccaf0a0.csv) |
| marker_atlas | overall | presence | [PNG](figures/atlas_overall_logical_connective_1_presence_4cbbc452.png) | [CSV](figures/atlas_overall_logical_connective_1_presence_4cbbc452.csv) |
| marker_atlas | justice | count | [PNG](figures/atlas_justice_reasoning_organization_1_count_ed808d1f.png) | [CSV](figures/atlas_justice_reasoning_organization_1_count_ed808d1f.csv) |
| marker_atlas | justice | density | [PNG](figures/atlas_justice_reasoning_organization_1_density_423d020b.png) | [CSV](figures/atlas_justice_reasoning_organization_1_density_423d020b.csv) |
| marker_atlas | justice | presence | [PNG](figures/atlas_justice_reasoning_organization_1_presence_dd06511b.png) | [CSV](figures/atlas_justice_reasoning_organization_1_presence_dd06511b.csv) |
| marker_atlas | justice | count | [PNG](figures/atlas_justice_epistemic_stance_1_count_94e1c84a.png) | [CSV](figures/atlas_justice_epistemic_stance_1_count_94e1c84a.csv) |
| marker_atlas | justice | density | [PNG](figures/atlas_justice_epistemic_stance_1_density_31f5dd5a.png) | [CSV](figures/atlas_justice_epistemic_stance_1_density_31f5dd5a.csv) |
| marker_atlas | justice | presence | [PNG](figures/atlas_justice_epistemic_stance_1_presence_5b101e86.png) | [CSV](figures/atlas_justice_epistemic_stance_1_presence_5b101e86.csv) |
| marker_atlas | justice | count | [PNG](figures/atlas_justice_logical_connective_1_count_205144e6.png) | [CSV](figures/atlas_justice_logical_connective_1_count_205144e6.csv) |
| marker_atlas | justice | density | [PNG](figures/atlas_justice_logical_connective_1_density_b822f66e.png) | [CSV](figures/atlas_justice_logical_connective_1_density_b822f66e.csv) |
| marker_atlas | justice | presence | [PNG](figures/atlas_justice_logical_connective_1_presence_75fe6be6.png) | [CSV](figures/atlas_justice_logical_connective_1_presence_75fe6be6.csv) |
| audit | overall |  | [PNG](figures/audit_dashboard_1c5fe0ef.png) | [CSV](figures/audit_dashboard_1c5fe0ef.csv) |
