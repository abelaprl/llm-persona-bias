$DATA = "D:\TugasAkhir\result\GMS8K\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b"
$PY   = "D:\TugasAkhir\.venv\Scripts\python.exe"
$REB  = "D:\TugasAkhir\src\rebuild"
$ACC  = "D:\TugasAkhir\src\analysis\calc_accuracy.py"
$OUT  = "D:\TugasAkhir\result\reports\GSM8K"
$S    = "rebuild_results_csv_gpt"
$PRE  = "deepseek_deepseek-r1-distill-qwen-32b_religion_persona"

$R = "$OUT\_tmp_$S"
New-Item -ItemType Directory -Force -Path $R | Out-Null

Write-Host "REBUILD: $S" -ForegroundColor Cyan
& $PY "$REB\rebuild_results_csv_gpt.py" --input_root $DATA --report_dir $R
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Rebuild failed (exit $LASTEXITCODE). Scroll up for Python error." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$R\results.csv")) {
    Write-Host "[ERROR] results.csv missing after rebuild." -ForegroundColor Red
    exit 1
}

Write-Host "ACCURACY: $S" -ForegroundColor Cyan
& $PY $ACC --csv_path "$R\results.csv" --out_dir $R --max_rows 1320
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] calc_accuracy failed." -ForegroundColor Red
    exit 1
}

Copy-Item "$R\results.csv"  "$OUT\${PRE}_${S}.csv"  -Force
Copy-Item "$R\accuracy.csv" "$OUT\${PRE}_${S}_accuracy.csv" -Force
Write-Host "Done: $S" -ForegroundColor Green
