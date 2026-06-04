$DATA = "D:\TugasAkhir\result\GMS8K\Religion_Persona\deepseek_deepseek-r1-distill-qwen-32b"
$PY   = "D:\TugasAkhir\.venv\Scripts\python.exe"
$REB  = "D:\TugasAkhir\src\rebuild"
$ACC  = "D:\TugasAkhir\src\analysis\calc_accuracy.py"
$OUT  = "D:\TugasAkhir\result\reports\GSM8K"
$PRE  = "deepseek_deepseek-r1-distill-qwen-32b_religion_persona"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$scripts = @(
    "rebuild_results_csv_gemini",
    "rebuild_results_csv_llama",
    "rebuild_results_csv_qwen",
    "rebuild_results_csv_gemma",
    "rebuild_results_csv_gemma9b",
    "rebuild_results_csv_gpt",
    "rebuild_results_csv_ministrail",
    "rebuild_results_csv_olmo"
)

foreach ($s in $scripts) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  RUNNING: $s" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $R = "$OUT\_tmp_$s"
    New-Item -ItemType Directory -Force -Path $R | Out-Null

    & $PY "$REB\$s.py" --input_root $DATA --report_dir $R
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] rebuild failed for $s" -ForegroundColor Red
        continue
    }

    & $PY $ACC --csv_path "$R\results.csv" --out_dir $R
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] calc_accuracy failed for $s" -ForegroundColor Red
        continue
    }

    Copy-Item "$R\results.csv"  "$OUT\${PRE}_${s}.csv"  -Force
    Copy-Item "$R\accuracy.csv" "$OUT\${PRE}_${s}_accuracy.csv" -Force

    Write-Host "  Done: ${PRE}_${s}.csv" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "  GENERATING COMPARISON TABLE" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
& $PY "D:\TugasAkhir\src\benchmark_rebuild_religion_persona.py"

Write-Host "`nAll done! Results in: $OUT" -ForegroundColor Green
