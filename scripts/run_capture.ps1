$ErrorActionPreference = "Stop"
. "$PSScriptRoot\use_utf8.ps1"

python -m huahua_assistant capture --image runtime\capture.png --debug runtime\debug.png
