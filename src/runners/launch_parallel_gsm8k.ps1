# Script untuk launch semua model secara parallel di terminal terpisah
# Usage: .\launch_parallel_gsm8k.ps1

$models = 31..49
$gsm8k_id = 1
$persona_id = 1

Write-Host "="*80
Write-Host "LAUNCHING $($models.Count) PARALLEL TERMINALS"
Write-Host "GSM8K ID: $gsm8k_id | Persona ID: $persona_id"
Write-Host "="*80
Write-Host ""

$launched = 0

foreach ($model_order in $models) {
    $command = "python src\run_gsm8k_single_model.py --model_order $model_order --gsm8k_id $gsm8k_id --persona_id $persona_id"
    
    # Launch new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; $command"
    
    $launched++
    Write-Host "[$launched/$($models.Count)] Launched terminal for Model Order $model_order" -ForegroundColor Green
    
    # Small delay to prevent overwhelming the system
    Start-Sleep -Milliseconds 300
}

Write-Host ""
Write-Host "="*80
Write-Host "✓ All $launched terminals launched!"
Write-Host "="*80
Write-Host ""
Write-Host "Terminals akan menutup otomatis setelah selesai."
Write-Host "Atau tekan CTRL+C di terminal mana pun untuk stop."
