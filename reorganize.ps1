$b = "E:\Tugas Akhir\TugasAkhir"
$s = "$b\src"
$errors = @()

function Mv($from, $to) {
    if (Test-Path $from) {
        Move-Item $from $to -Force
    } else {
        $script:errors += "NOT FOUND: $from"
    }
}

# ────────────────────────────────────────────────────
# src/runners/   – scripts to run experiments
# ────────────────────────────────────────────────────
Mv "$s\batch_explicit_runner.py"               "$s\runners\batch_explicit_runner.py"
Mv "$s\run_batch_gsm8k.py"                     "$s\runners\run_batch_gsm8k.py"
Mv "$s\run_gsm8k_batch.py"                     "$s\runners\run_gsm8k_batch.py"
Mv "$s\run_gsm8k_scenario_a.py"                "$s\runners\run_gsm8k_scenario_a.py"
Mv "$s\run_openrouter.py"                      "$s\runners\run_openrouter.py"
Mv "$s\run_warmup_man_implicit.py"             "$s\runners\run_warmup_man_implicit.py"
Mv "$s\run_warmup_woman_implicit.py"           "$s\runners\run_warmup_woman_implicit.py"
Mv "$s\gsm8k_scenario_a.py"                    "$s\runners\gsm8k_scenario_a.py"
Mv "$s\groq_woman_implicits_warmup_correct.py" "$s\runners\groq_woman_implicits_warmup_correct.py"
Mv "$b\continue_groq_experiment.py"            "$s\runners\continue_groq_experiment.py"
Mv "$b\check_warmup_status.py"                 "$s\runners\check_warmup_status.py"
Mv "$b\test_explicit_man.py"                   "$s\runners\test_explicit_man.py"
Mv "$b\launch_parallel_gsm8k.ps1"             "$s\runners\launch_parallel_gsm8k.ps1"

# ────────────────────────────────────────────────────
# src/rerun/   – rerun failed / truncated cases
# ────────────────────────────────────────────────────
Mv "$b\rerun_api_olmo.py"                      "$s\rerun\rerun_api_olmo.py"
Mv "$b\rerun_empty_model31.py"                 "$s\rerun\rerun_empty_model31.py"
Mv "$b\rerun_gsm8k_schema_preserve.py"         "$s\rerun\rerun_gsm8k_schema_preserve.py"
Mv "$b\rerun_gsm8k_still_wrong.py"             "$s\rerun\rerun_gsm8k_still_wrong.py"
Mv "$b\update_rerun_json_csv.py"               "$s\rerun\update_rerun_json_csv.py"
Mv "$b\audit_rerun_to_results_fixed.py"        "$s\rerun\audit_rerun_to_results_fixed.py"
Mv "$b\scripts\rerun_blank_extracted_olmo32b.py" "$s\rerun\rerun_blank_extracted_olmo32b.py"
Mv "$b\scripts\merge_rerun_into_result.py"     "$s\rerun\merge_rerun_into_result.py"
Mv "$b\scripts\generate_rerun_plan.py"         "$s\rerun\generate_rerun_plan.py"

# ────────────────────────────────────────────────────
# src/analysis/  – accuracy, mismatch, wrong-answer analysis
# ────────────────────────────────────────────────────
Mv "$b\analyze_wrong_answers.py"               "$s\analysis\analyze_wrong_answers.py"
Mv "$b\analyze_and_fix_extraction.py"          "$s\analysis\analyze_and_fix_extraction.py"
Mv "$b\compute_accuracy_from_results_csv.py"   "$s\analysis\compute_accuracy_from_results_csv.py"
Mv "$b\compute_accuracy_openai_gpt_oss_20b.py" "$s\analysis\compute_accuracy_openai_gpt_oss_20b.py"
Mv "$b\scripts\analyze_apply_report.py"        "$s\analysis\analyze_apply_report.py"
Mv "$b\scripts\analyze_no_extract.py"          "$s\analysis\analyze_no_extract.py"
Mv "$b\scripts\analyze_remaining.py"           "$s\analysis\analyze_remaining.py"
Mv "$b\scripts\audit_blank_extracted_answer.py" "$s\analysis\audit_blank_extracted_answer.py"
Mv "$b\scripts\audit_wrong_answers_openai_gpt-oss-20b.py" "$s\analysis\audit_wrong_answers_openai_gpt-oss-20b.py"
Mv "$b\scripts\categorize_mismatches_model34.py" "$s\analysis\categorize_mismatches_model34.py"
Mv "$b\scripts\categorize_wrong_model34.py"    "$s\analysis\categorize_wrong_model34.py"
Mv "$b\scripts\dump_all_wrong_full_responses.py" "$s\analysis\dump_all_wrong_full_responses.py"
Mv "$b\scripts\dump_mismatches_to_csv.py"      "$s\analysis\dump_mismatches_to_csv.py"
Mv "$b\scripts\export_mismatches_review_examples.py" "$s\analysis\export_mismatches_review_examples.py"
Mv "$b\scripts\make_accuracy_csv.py"           "$s\analysis\make_accuracy_csv.py"
Mv "$s\analyze_results.py"                     "$s\analysis\analyze_results.py"
Mv "$s\generate_analysis_csv.py"               "$s\analysis\generate_analysis_csv.py"
Mv "$s\generate_model_summary.py"              "$s\analysis\generate_model_summary.py"
Mv "$s\generate_reports_by_model_persona.py"   "$s\analysis\generate_reports_by_model_persona.py"

# ────────────────────────────────────────────────────
# src/extract/   – extraction logic & debug extraction
# ────────────────────────────────────────────────────
Mv "$b\extractor34.py"                         "$s\extract\extractor34.py"
Mv "$b\debug_norm.py"                          "$s\extract\debug_norm.py"
Mv "$b\show_extraction_errors.py"              "$s\extract\show_extraction_errors.py"
Mv "$b\show_extraction_errors_full.py"         "$s\extract\show_extraction_errors_full.py"
Mv "$b\analyze_mismatch.py"                    "$s\extract\analyze_mismatch.py"
Mv "$s\extractor_fix.py"                       "$s\extract\extractor_fix.py"
Mv "$s\debug_parser.py"                        "$s\extract\debug_parser.py"
Mv "$s\debug_parser_v2.py"                     "$s\extract\debug_parser_v2.py"
Mv "$b\scripts\reextract_all_model34.py"       "$s\extract\reextract_all_model34.py"
Mv "$b\scripts\reextract_and_analyze.py"       "$s\extract\reextract_and_analyze.py"
Mv "$b\scripts\test_extract_final_answer.py"   "$s\extract\test_extract_final_answer.py"

# ────────────────────────────────────────────────────
# src/reports/   – generate CSV / Excel output
# ────────────────────────────────────────────────────
Mv "$b\make_wrong_answers_csv.py"              "$s\reports\make_wrong_answers_csv.py"
Mv "$b\make_wrong_answers_only.py"             "$s\reports\make_wrong_answers_only.py"
Mv "$b\make_wrong_answers_samples.py"          "$s\reports\make_wrong_answers_samples.py"
Mv "$b\make_wrong_answers_smart_extract.py"    "$s\reports\make_wrong_answers_smart_extract.py"
Mv "$s\generate_csv_results.py"                "$s\reports\generate_csv_results.py"
Mv "$s\generate_excel.py"                      "$s\reports\generate_excel.py"
Mv "$s\generate_results_csv.py"                "$s\reports\generate_results_csv.py"
Mv "$b\scripts\split_persona_results.py"       "$s\reports\split_persona_results.py"
Mv "$b\scripts\split_results_per_persona.py"   "$s\reports\split_results_per_persona.py"
Mv "$b\scripts\rebuild_openai_gpt_oss_20b_results.py" "$s\reports\rebuild_openai_gpt_oss_20b_results.py"

# ────────────────────────────────────────────────────
# src/inspect/   – count, inspect, find data issues
# ────────────────────────────────────────────────────
Mv "$b\check_noextract.py"                     "$s\inspect\check_noextract.py"
Mv "$b\count_empty.py"                         "$s\inspect\count_empty.py"
Mv "$b\generate_implicit_summary.py"           "$s\inspect\generate_implicit_summary.py"
Mv "$b\generate_commands.ps1"                  "$s\inspect\generate_commands.ps1"
Mv "$b\scripts\count_empty_model34.py"         "$s\inspect\count_empty_model34.py"
Mv "$b\scripts\count_truncated_model34.py"     "$s\inspect\count_truncated_model34.py"
Mv "$b\scripts\find_truncated_responses.py"    "$s\inspect\find_truncated_responses.py"
Mv "$b\scripts\correlate_truncated_with_tokens.py" "$s\inspect\correlate_truncated_with_tokens.py"
Mv "$b\scripts\inspect_token.py"               "$s\inspect\inspect_token.py"
Mv "$b\scripts\print_candidates.py"            "$s\inspect\print_candidates.py"

# ────────────────────────────────────────────────────
# src/parse/ — parsing utilities
# ────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "$s\parse" | Out-Null
Mv "$s\parse_gsm8k_results.py"                 "$s\parse\parse_gsm8k_results.py"
Mv "$s\checkpoint_monitor.py"                  "$s\parse\checkpoint_monitor.py"

# ────────────────────────────────────────────────────
# Remaining root scripts not categorised above
# ────────────────────────────────────────────────────
# analyze_model scripts – analysis
foreach ($f in @("analyze_model31.py","analyze_model32.py","analyze_model33.py","analyze_model34.py")) {
    Mv "$b\$f" "$s\analysis\$f"
}
Mv "$b\rerun_empty_model34.py"                 "$s\rerun\rerun_empty_model34.py"

# scripts/ leftovers
Mv "$b\scripts\analyze_apply_report.py"        "$s\analysis\analyze_apply_report.py"

# duplicate scripts/rebuild_results_csv.py already moved as core (if still exists, skip)
if (Test-Path "$b\scripts\rebuild_results_csv.py") {
    Mv "$b\scripts\rebuild_results_csv.py"     "$s\reports\rebuild_results_csv_scripts.py"
}

# ────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────
Write-Host "`n=== DONE ==="
if ($errors.Count -gt 0) {
    Write-Host "`nSkipped (not found):"
    $errors | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "All files moved successfully."
}
