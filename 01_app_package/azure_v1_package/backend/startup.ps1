$ErrorActionPreference = "Stop"

$appRoot = $env:APP_ROOT
if ([string]::IsNullOrWhiteSpace($appRoot)) {
  $appRoot = "D:\\home\\site\\wwwroot\\app"
}

if ([string]::IsNullOrWhiteSpace($env:PORT)) {
  $env:PORT = "8000"
}

if ([string]::IsNullOrWhiteSpace($env:IWERP_BASE_DIR)) {
  $env:IWERP_BASE_DIR = $appRoot
}

$env:PYTHONPATH = "$appRoot\backend;$appRoot;$env:PYTHONPATH"
Set-Location $appRoot
python -m uvicorn backend.main:app --host 0.0.0.0 --port $env:PORT --workers 1
