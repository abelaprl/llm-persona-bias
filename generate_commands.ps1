# Script ini generate command untuk jalankan 19 model secara parallel
# Setiap model akan test GSM8K ID 1 dengan persona ID 1

$models = 31..49
$gsm8k_id = 1
$persona_id = 1

Write-Host "="*80
Write-Host "PARALLEL EXECUTION COMMANDS - GSM8K ID $gsm8k_id with Persona $persona_id"
Write-Host "="*80
Write-Host ""
Write-Host "Pilih salah satu cara:"
Write-Host ""
Write-Host "--- CARA 1: AUTOMATIC (Buka semua terminal otomatis) ---"
Write-Host "Jalankan command ini di PowerShell:"
Write-Host ""
Write-Host ".\launch_parallel_gsm8k.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host ""
Write-Host "--- CARA 2: MANUAL (Copy-paste satu per satu di terminal berbeda) ---"
Write-Host "Buka terminal baru untuk setiap command ini:"
Write-Host ""

foreach ($model_order in $models) {
    $command = "python src\run_gsm8k_single_model.py --model_order $model_order --gsm8k_id $gsm8k_id --persona_id $persona_id"
    Write-Host "Terminal $($model_order - 30): $command" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "="*80
Write-Host "Total: $($models.Count) terminals"
Write-Host "="*80
