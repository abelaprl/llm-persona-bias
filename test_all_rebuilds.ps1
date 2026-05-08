# Test semua rebuild scripts untuk model 44 & 45

$scripts = @(
    @{name="qwen"; script="rebuild_results_csv_qwen.py"; folder="arcee-ai_trinity-large-thinking_qwen"},
    @{name="llama"; script="rebuild_results_csv_llama.py"; folder="arcee-ai_trinity-large-thinking_llama"},
    @{name="gemma9b"; script="rebuild_results_csv_gemma9b.py"; folder="arcee-ai_trinity-large-thinking_gemma9b"},
    @{name="gemma"; script="rebuild_results_csv_gemma.py"; folder="arcee-ai_trinity-large-thinking_gemma"},
    @{name="gpt"; script="rebuild_results_csv_gpt.py"; folder="arcee-ai_trinity-large-thinking_gpt"},
    @{name="gemini"; script="rebuild_results_csv_gemini.py"; folder="arcee-ai_trinity-large-thinking_gemini"},
    @{name="olmo"; script="rebuild_results_csv_olmo.py"; folder="arcee-ai_trinity-large-thinking_olmo"},
    @{name="ministrail"; script="rebuild_results_csv_ministrail.py"; folder="arcee-ai_trinity-large-thinking_ministrail"}
)

$results = @()

foreach ($s in $scripts) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $($s.name)" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan

    # Update script paths
    $scriptPath = "src/$($s.script)"

    # Update input/output paths in script
    $content = Get-Content $scriptPath -Raw
    $content = $content -replace "result\\[^'`"]*arcee-ai", "result\GMS8K\arcee-ai"
    $content = $content -replace "reports\\[^'`"]*arcee-ai[^'`"]*", "reports\$($s.folder)"
    Set-Content $scriptPath $content

    # Run rebuild
    Write-Host "[1/2] Rebuilding CSV with $($s.name) extraction..." -ForegroundColor Green
    $rebuild = python $scriptPath 2>&1 | tail -20
    Write-Host $rebuild

    # Update calc_accuracy paths
    $calcPath = "src/calc_accuracy.py"
    $calcContent = Get-Content $calcPath -Raw
    $calcContent = $calcContent -replace "reports\\[^'`"]*", "reports\$($s.folder)"
    Set-Content $calcPath $calcContent

    # Run calc_accuracy
    Write-Host "`n[2/2] Calculating accuracy with $($s.name)..." -ForegroundColor Green
    $accuracy = python $calcPath 2>&1

    # Extract overall accuracy
    $overallLine = $accuracy | Select-String "OVERALL"
    Write-Host $overallLine

    # Parse accuracy from output
    if ($overallLine -match "(\d+\.\d+)%") {
        $acc = [float]$matches[1]
        $results += @{name=$s.name; accuracy=$acc; output=$accuracy}
    }
}

# Print final comparison
Write-Host "`n`n" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan
Write-Host "FINAL COMPARISON - MODEL 44 (ARCEE AI)" -ForegroundColor Yellow
Write-Host "="*60 -ForegroundColor Cyan

$sorted = $results | Sort-Object accuracy -Descending
$sorted | ForEach-Object {
    Write-Host "$($_.name.PadRight(15)) | Accuracy: $($_.accuracy)%" -ForegroundColor $(if ($_.accuracy -gt 90) { "Green" } elseif ($_.accuracy -gt 85) { "Yellow" } else { "Red" })
}

Write-Host "`nBEST: $($sorted[0].name) with $($sorted[0].accuracy)%" -ForegroundColor Green
Write-Host "WORST: $($sorted[-1].name) with $($sorted[-1].accuracy)%" -ForegroundColor Red