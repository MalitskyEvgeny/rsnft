<#
.SYNOPSIS
    Прямая задача + HTML-отчёт (report_full.jl).

.DESCRIPTION
    Основной режим — АДАПТИВНАЯ СЕТКА (adaptive_basin_fvm.jl). Флаг -Uniform
    переключает на классический равномерный солвер (basin_fvm.jl).

.EXAMPLE
    .\run_forward.ps1                                         # адаптивная сетка
    .\run_forward.ps1 -AdaptiveBase 30,60 -AdaptiveLevel 1
    .\run_forward.ps1 -Uniform -Nx 30 -Nz 60 -Force
    .\run_forward.ps1 -DataDir "C:\путь\к\данным" -OpenReport
#>
[CmdletBinding()]
param(
    [string]$DataDir       = "",
    [string]$AdaptiveBase  = "16,32",
    [int]   $AdaptiveLevel = 2,
    [switch]$Uniform,
    [int]   $Nx            = 60,
    [int]   $Nz            = 120,
    [double]$MaxDtMa       = 2.5,
    [int]   $PicardIters   = 3,
    [string]$Out           = "",
    [switch]$Force,
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$R3 = Resolve-Path (Join-Path $ScriptDir "..")
$RepoRoot = Resolve-Path (Join-Path $R3 "..")
$JuliaExe = if ($env:JULIA) { $env:JULIA } else { "julia" }

# "Данные_для_прямой_задачи" посимвольно, чтобы не зависеть от кодировки файла
$DirectName = -join [char[]](0x0414,0x0430,0x043D,0x043D,0x044B,0x0435,0x005F,
    0x0434,0x043B,0x044F,0x005F,0x043F,0x0440,0x044F,0x043C,0x043E,0x0439,
    0x005F,0x0437,0x0430,0x0434,0x0430,0x0447,0x0438)

if (-not $DataDir) { $DataDir = Join-Path $RepoRoot (Join-Path $DirectName "data1") }
if (-not (Test-Path $DataDir)) { throw "data dir not found: $DataDir" }

$TS = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $Out) { $Out = Join-Path $RepoRoot "results\report_forward_$TS" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

if ($Uniform) {
    Write-Host "=== forward (uniform): $(Split-Path -Leaf $DataDir), grid ${Nx}x${Nz} ==="
    $SolverScript = Join-Path $R3 "basin_fvm.jl"
    $SolverArgs = @("--data-dir", $DataDir, "--cache-dir", (Join-Path $RepoRoot "results"),
                    "--nx", "$Nx", "--nz", "$Nz", "--max-dt-ma", "$MaxDtMa",
                    "--picard-iterations", "$PicardIters")
} else {
    Write-Host "=== forward (adaptive): $(Split-Path -Leaf $DataDir), base $AdaptiveBase, level $AdaptiveLevel ==="
    $SolverScript = Join-Path $R3 "adaptive_basin_fvm.jl"
    $SolverArgs = @("--data-dir", $DataDir, "--cache-dir", (Join-Path $RepoRoot "results"),
                    "--base-nx", ($AdaptiveBase -split ',')[0], "--base-nz", ($AdaptiveBase -split ',')[1],
                    "--max-level", "$AdaptiveLevel", "--max-dt-ma", "$MaxDtMa",
                    "--picard-iterations", "$PicardIters")
}
if ($Force) { $SolverArgs += "--force" }

$SolverOut = & $JuliaExe "--project=$R3" $SolverScript @SolverArgs 2>&1 | Tee-Object -Variable AllOut
$RunLine = ($AllOut | Select-String -Pattern "^(RUN|CACHE_HIT)" | Select-Object -Last 1)
if (-not $RunLine) { throw "solver did not report a run id" }
$RunId = (($RunLine.ToString() -replace '[^\w\s]', ' ') -split '\s+' |
          Where-Object { $_ -match '^[0-9a-fA-F]{8,}$' } | Select-Object -First 1)
if (-not $RunId) { throw "could not parse run id from: $($RunLine.ToString())" }
$RunDir = Join-Path $RepoRoot "results\$RunId"
Write-Host "run dir: $RunDir"

Write-Host "=== report ==="
& $JuliaExe "--project=$R3" (Join-Path $R3 "report_full.jl") --direct $RunDir --out (Join-Path $Out "report.html")
if ($LASTEXITCODE -ne 0) { throw "report_full.jl failed" }

Write-Host "=== summary ==="
Write-Host "report   : $(Join-Path $Out 'report.html')"
Write-Host "run dir  : $RunDir"
$Report = Join-Path $Out "report.html"
if ($OpenReport -and (Test-Path $Report)) { Start-Process $Report }
