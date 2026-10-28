param(
    [ValidateSet('repair', 'aami', 'repair-aami', 'indices')]
    [string]$Profile = 'repair-aami',
    [ValidateRange(1, 4)]
    [int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$stateDirectory = Join-Path $projectRoot '.local'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the isolated project venv first.'
}
New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
$processFile = Join-Path $stateDirectory 'data-transfer-process.json'
if (Test-Path -LiteralPath $processFile) {
    $previous = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $active = Get-CimInstance Win32_Process -Filter "ProcessId = $($previous.process_id)"
    if ($active -and $active.CommandLine -like '*sync_pulsedb.py*') {
        throw "A data repair process is already active: $($previous.process_id)"
    }
}
$scriptPath = Join-Path $PSScriptRoot 'sync_pulsedb.py'
$arguments = @('-u', ('"{0}"' -f $scriptPath), '--profile', $Profile, '--workers', $Workers)
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $stateDirectory 'data-transfer.log') `
    -RedirectStandardError (Join-Path $stateDirectory 'data-transfer.err.log')
@{process_id=$process.Id; profile=$Profile; started_at=(Get-Date).ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
Write-Output "Data repair started in background. PID=$($process.Id)"
Write-Output 'Inspect .local/data-transfer.log or run .venv\Scripts\python.exe scripts/data_status.py'
