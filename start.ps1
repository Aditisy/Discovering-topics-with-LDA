$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Topic Atlas: http://127.0.0.1:8784'
python -m http.server 8784 --bind 127.0.0.1 --directory dist
