param(
    [string]$RunName = "pilot-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [int]$Workers = 3
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv. Create it and install the project before running the pilot."
}

& $python scripts/run_experiment.py --config configs/pilot.json --workers $Workers --run-name $RunName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/validate_results.py "results/pilot/$RunName" --config configs/pilot.json
exit $LASTEXITCODE
