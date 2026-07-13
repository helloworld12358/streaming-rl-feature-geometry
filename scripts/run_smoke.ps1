param(
    [string]$RunName = "smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv. Create it and install the project before running smoke."
}

& $python scripts/run_experiment.py --config configs/smoke.json --workers $Workers --run-name $RunName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/validate_results.py "results/smoke/$RunName" --config configs/smoke.json
exit $LASTEXITCODE
