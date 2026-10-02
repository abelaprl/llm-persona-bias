@echo off
cd /d D:\TugasAkhir
python -u -m src.core.linguistic_markers.gender_dataset_audit --extraction-workers 6 --output-dir "D:\TugasAkhir\result\ETHICS\reports\Gender_Persona\linguistic_markers_gender_candidate_20260821_v1_6_0" 1>"D:\TugasAkhir\v16_full_run_20260822_final_stdout.log" 2>"D:\TugasAkhir\v16_full_run_20260822_final_stderr.log"
