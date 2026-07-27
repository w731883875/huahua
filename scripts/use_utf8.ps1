$script:HuahuaUtf8NoBom = New-Object System.Text.UTF8Encoding($false)

[Console]::InputEncoding = $script:HuahuaUtf8NoBom
[Console]::OutputEncoding = $script:HuahuaUtf8NoBom
$OutputEncoding = $script:HuahuaUtf8NoBom

$isWindowsHost = ($env:OS -eq "Windows_NT") -or ((Get-Variable IsWindows -ErrorAction SilentlyContinue) -and $IsWindows)
if ($isWindowsHost) {
    try {
        chcp.com 65001 > $null
    } catch {
        Write-Warning "Unable to switch console code page to UTF-8: $($_.Exception.Message)"
    }
}

function Write-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $resolvedParent = Split-Path -Parent $Path
    if ($resolvedParent) {
        New-Item -ItemType Directory -Force -Path $resolvedParent | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Value, $script:HuahuaUtf8NoBom)
}
