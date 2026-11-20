<#
.SYNOPSIS
    用**项目解释器 + 锁文件**打一份可追溯的 exe。

.DESCRIPTION
    以前这里只有两行：装一遍范围版的依赖文件，再调一次 PyInstaller。两次用的
    都是 PATH 上随手拿到的那个 python，依赖是范围版（>=），装失败也没人看退出码。
    结果就是"打包成功、采集退回截屏"，而报告里只有一句 Built。

    现在做四件事：
    1. 显式用 `.venv\Scripts\python.exe`（可用 -Python 覆盖），不用 PATH 上的；
    2. 从 `requirements-lock.txt` 精确安装（不再用范围版 requirements.txt）；
    3. 每一步都检查退出码，失败立刻停，不会打印 Built；
    4. 产物写到独立目录 `build_artifacts\<时间戳>\`，旁边放报告：
       代码版本、工作区改动摘要、锁文件摘要、exe 哈希。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -ValidateOnly
#>
param(
    # 指定解释器。默认用项目自己的 .venv，绝不用 PATH 上的 python。
    [string]$Python = '',
    # 产物根目录。默认 build_artifacts\<时间戳>。
    [string]$OutRoot = '',
    # 只验证环境与依赖，不真的调 PyInstaller（CI 预检、无打包需求时用）。
    [switch]$ValidateOnly,
    # 跳过 pip install（依赖已经装好时省时间）。
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Title 失败（退出码 $LASTEXITCODE）"
    }
}

# ---- 1. 解释器 ------------------------------------------------------------
if (-not $Python) {
    $candidate = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $candidate)) {
        throw "找不到项目解释器：$candidate。先建好 .venv，或用 -Python 指定。"
    }
    $Python = $candidate
}
if (-not (Test-Path $Python)) {
    throw "解释器不存在：$Python"
}

# ---- 2. 产物目录 ----------------------------------------------------------
if (-not $OutRoot) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutRoot = Join-Path $projectRoot "build_artifacts\$stamp"
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Push-Location $projectRoot
try {
    # ---- 3. 依赖：锁文件精确安装 ---------------------------------------
    if (-not $SkipInstall) {
        Write-Host "==> 从 requirements-lock.txt 安装依赖（$Python）"
        Invoke-Checked 'pip install' {
            & $Python -m pip install -r requirements-lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        }
    }

    # ---- 4. 环境检查（包存在 / 模块可导入 / 锁一致） --------------------
    Write-Host '==> 环境检查'
    Invoke-Checked '环境检查' { & $Python -B scripts\check_environment.py }

    $built = $false
    if (-not $ValidateOnly) {
        Write-Host "==> PyInstaller（产物 -> $OutRoot）"
        Invoke-Checked 'PyInstaller' {
            & $Python -m PyInstaller --noconfirm `
                --distpath $OutRoot `
                --workpath (Join-Path $OutRoot 'work') `
                HuahuaAssistant.spec
        }
        $built = $true
    }

    # ---- 5. 报告 ------------------------------------------------------
    $exePath = Join-Path $OutRoot 'HuahuaAssistant.exe'
    $exeHash = $null
    $exeSize = $null
    if (Test-Path $exePath) {
        $exeHash = (Get-FileHash -Algorithm SHA256 -Path $exePath).Hash
        $exeSize = (Get-Item $exePath).Length
    }
    elseif ($built) {
        throw "PyInstaller 报了成功，但没找到 $exePath"
    }

    $gitHead = (& git rev-parse HEAD 2>$null)
    $gitShort = (& git rev-parse --short HEAD 2>$null)
    $gitBranch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    $gitStatus = @(& git status --porcelain 2>$null)
    # build/ dist/ runtime/ 也在版本库里（历史遗留），它们每跑一次就变，
    # 直接数会让"工作区改了 648 个文件"这种读数失去意义。分成两组看。
    $noise = @('build/', 'dist/', 'runtime/', 'build_artifacts/')
    $sourceChanges = @($gitStatus | Where-Object {
            $line = $_.Substring([Math]::Min(3, $_.Length)).Trim('"')
            -not ($noise | Where-Object { $line -like "$_*" })
        })
    $lockPath = Join-Path $projectRoot 'requirements-lock.txt'
    $lockHash = (Get-FileHash -Algorithm SHA256 -Path $lockPath).Hash

    $report = [ordered]@{
        schema      = 'huahua-build-report-v1'
        built       = $built
        validate_only = [bool]$ValidateOnly
        timestamp   = (Get-Date).ToString('o')
        python      = (& $Python -c 'import sys; print(sys.version.split()[0])')
        python_path = $Python
        git         = [ordered]@{
            head            = "$gitHead".Trim()
            short           = "$gitShort".Trim()
            branch          = "$gitBranch".Trim()
            dirty_files     = $gitStatus.Count
            dirty_source    = $sourceChanges.Count
            dirty_summary   = @($sourceChanges | Select-Object -First 40)
            dirty_truncated = ($sourceChanges.Count -gt 40)
        }
        lock        = [ordered]@{
            path   = 'requirements-lock.txt'
            sha256 = $lockHash
            # 必须按 UTF-8 读：默认编码会把中文注释读成乱码，还会把几行粘成一条。
            entries = @(Get-Content -Path $lockPath -Encoding UTF8 |
                Where-Object { $_ -match '==' } |
                ForEach-Object { $_.Trim() })
        }
        exe         = [ordered]@{
            path   = if ($exeHash) { $exePath } else { $null }
            sha256 = $exeHash
            bytes  = $exeSize
        }
        artifact_dir = $OutRoot
    }
    $reportPath = Join-Path $OutRoot 'build_report.json'
    # 不带 BOM 写：带 BOM 的 JSON 会让一部分严格解析器直接失败。
    Write-Utf8NoBomFile -Path $reportPath -Value ($report | ConvertTo-Json -Depth 6)

    Write-Host ''
    if ($built) {
        Write-Host "Built: $exePath"
        Write-Host "SHA256: $exeHash"
    }
    else {
        Write-Host 'ValidateOnly：环境与依赖已通过，未打包。'
    }
    Write-Host "报告: $reportPath"
}
finally {
    Pop-Location
}
