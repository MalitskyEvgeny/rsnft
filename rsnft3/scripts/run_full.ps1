<#
.SYNOPSIS
    Прямая + обратная задача + единый HTML-отчёт (report_full.jl).

.DESCRIPTION
    Основной режим — АДАПТИВНАЯ СЕТКА:
      * прямой прогон — adaptive_basin_fvm.jl (-Base/-Level);
      * инверсия — контроль на адаптивной сетке (run_inverse.jl -AdaptiveFinal).
    Флаг -Uniform переключает всё на классические равномерные сетки.

.EXAMPLE
    .\run_full.ps1 -Grid 30,60 -DeBudget 200            # быстрый контрольный прогон
    .\run_full.ps1 -Twin -Q 75,80,85,90                 # twin-режим
    .\run_full.ps1 -Uniform -Grid 30,60 -Nx 30 -Nz 60   # равномерные сетки
    .\run_full.ps1 -OpenReport                          # открыть отчёт после прогона
#>
[CmdletBinding()]
param(
    [string]$DataDir       = "",
    [string]$InvDataDir    = "",
    [string]$Wells         = "",
    [switch]$Twin,
    [string]$Q             = "75,80,85,90",
    [int]   $TwinSeed      = 42,
    [string]$Grid          = "30,60",
    [string]$Base          = "16,32",
    [int]   $Level         = 2,
    [string]$AdaptiveBase  = "16,32",
    [int]   $AdaptiveLevel = 2,
    [switch]$Uniform,
    [int]   $Nx            = 30,
    [int]   $Nz            = 60,
    [int]   $Ensemble      = 0,   # 0 = авто: по числу логических ядер
    [int]   $Workers       = 0,   # 0 = авто: по числу ансамблей
    [int]   $DeBudget      = 2000,
    [int]   $Threads       = 8,
    [int]   $NmIterations  = 200,
    [double]$SigmaT        = 2.0,
    [double]$SigmaRo       = 0.05,
    [double]$Reg           = 0.3,
    [string]$NodeXs        = "",
    [string]$Epochs        = "",
    [int]   $Seed          = 1,
    [string]$Theta0File    = "",
    [string]$Out           = "",
    [switch]$Force,
    [switch]$OpenReport,
    [switch]$QScalar,
    [switch]$NoQScalar
)

# UTF-8 консоль (кириллица в логах без '?????')
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null
} catch { }


$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$R3 = Resolve-Path (Join-Path $ScriptDir "..")
$RepoRoot = Resolve-Path (Join-Path $R3 "..")
$JuliaExe = if ($env:JULIA) { $env:JULIA } else { "julia" }

# Требование версии: Manifest зафиксирован для Julia 1.13.0 (проверено на 1.13.0).
$JuliaVerOut = & $JuliaExe --version 2>$null
if ($LASTEXITCODE -ne 0 -or -not $JuliaVerOut) { throw "Julia not found: '$JuliaExe'." }
$JuliaVer = ("$JuliaVerOut" -replace '^julia version ', '').Trim()
if ([version]$JuliaVer -lt [version]"1.13.0") {
    throw ("Требуется Julia >= 1.13 (Manifest зафиксирован для 1.13.0); найдено: {0}" -f $JuliaVer)
}

# "Данные_для_прямой_задачи" и "Данные для обратной задачи" посимвольно,
# чтобы не зависеть от кодировки файла
$DirectName = -join [char[]](0x0414,0x0430,0x043D,0x043D,0x044B,0x0435,0x005F,
    0x0434,0x043B,0x044F,0x005F,0x043F,0x0440,0x044F,0x043C,0x043E,0x0439,
    0x005F,0x0437,0x0430,0x0434,0x0430,0x0447,0x0438)
$InverseName = -join [char[]](0x0414,0x0430,0x043D,0x043D,0x044B,0x0435,0x0020,
    0x0434,0x043B,0x044F,0x0020,0x043E,0x0431,0x0440,0x0430,0x0442,0x043D,0x043E,0x0439,
    0x0020,0x0437,0x0430,0x0434,0x0430,0x0447,0x0438)

if (-not $DataDir)    { $DataDir = Join-Path $RepoRoot (Join-Path $DirectName "data1") }
if (-not $InvDataDir) { $InvDataDir = Join-Path $RepoRoot $InverseName }
if (-not (Test-Path $DataDir))    { throw "data dir not found: $DataDir" }
if (-not (Test-Path $InvDataDir)) { throw "inverse data dir not found: $InvDataDir" }
if (-not $Wells) {
    $Wells = (Join-Path $InvDataDir "TemperatureWellData.xlsx") + ";" +
             (Join-Path $InvDataDir "VitriniteWellData.xlsx")
}
if ($Theta0File -and -not (Test-Path $Theta0File)) { throw "theta0-file not found: $Theta0File" }

# Ранние диагностируемые проверки вместо молчаливого "solver did not report a run id"
if (-not (Get-Command $JuliaExe -ErrorAction SilentlyContinue)) {
    throw "Julia not found: '$JuliaExe'. Install Julia 1.9+ or set JULIA env var"
}
if (-not (Test-Path $DataDir))    { throw "data dir not found: $DataDir (unzip data folders next to rsnft3 or pass -DataDir)" }
if (-not (Test-Path $InvDataDir)) { throw "inverse data dir not found: $InvDataDir (pass -InvDataDir)" }

$TS = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $Out) { $Out = Join-Path $RepoRoot "results\report_full_$TS" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

if ($Uniform) {
    Write-Host "=== Stage 1/4: forward solve (uniform): $(Split-Path -Leaf $DataDir), grid ${Nx}x${Nz} ==="
    $SolverScript = Join-Path $R3 "basin_fvm.jl"
    $SolverArgs = @("--data-dir", $DataDir, "--cache-dir", (Join-Path $RepoRoot "results"),
                    "--nx", "$Nx", "--nz", "$Nz")
} else {
    Write-Host "=== Stage 1/4: forward solve (adaptive): $(Split-Path -Leaf $DataDir), base $Base, level $Level ==="
    $SolverScript = Join-Path $R3 "adaptive_basin_fvm.jl"
    $SolverArgs = @("--data-dir", $DataDir, "--cache-dir", (Join-Path $RepoRoot "results"),
                    "--base-nx", ($Base -split ',')[0], "--base-nz", ($Base -split ',')[1],
                    "--max-level", "$Level")
}
if ($Force) { $SolverArgs += "--force" }
# PS5: stderr native-processa pri 2>&1 stanovitsya ErrorRecord - EAP=Stop ubivaet skript
$ErrorActionPreference = "Continue"
$SolverOut = & $JuliaExe "--project=$R3" "-t" "$Threads" $SolverScript @SolverArgs 2>&1 | Tee-Object -Variable AllOut
$ErrorActionPreference = "Stop"
$RunLine = ($AllOut | Select-String -Pattern "^(RUN|CACHE_HIT)" | Select-Object -Last 1)
if (-not $RunLine) {
    Write-Host "=== solver output (failure diagnostics) ===" -ForegroundColor Yellow
    $AllOut | ForEach-Object { Write-Host $_ }
    throw "solver did not report a run id"
}
$RunId = (($RunLine.ToString() -replace '[^\w\s]', ' ') -split '\s+' |
          Where-Object { $_ -match '^[0-9a-fA-F]{8,}$' } | Select-Object -First 1)
if (-not $RunId) { throw "could not parse run id from: $($RunLine.ToString())" }
$RunDir = Join-Path $RepoRoot "results\$RunId"
Write-Host "run dir: $RunDir"

Write-Host "=== Stage 2/4: inversion ==="
$WellsForInverse = $Wells
if ($Twin) {
    $TwinFile = Join-Path $Out "twin_grid.json"
    Write-Host "--- twin: q=$Q, grid $Grid ---"
    & $JuliaExe "--project=$R3" "-t" "$Threads" (Join-Path $R3 "make_twin.jl") `
        --data-dir $InvDataDir --wells ($Wells -split ';')[0] --grid $Grid --q $Q `
        --sigma-T $SigmaT --sigma-Ro $SigmaRo --seed $TwinSeed --out $TwinFile
    if ($LASTEXITCODE -ne 0) { throw "make_twin.jl failed" }
    $WellsForInverse = $TwinFile
}
$InverseArgs = @("--project=$R3", "-t", "$Threads", (Join-Path $R3 "run_inverse.jl"),
                 "--data-dir", $InvDataDir, "--wells", $WellsForInverse,
                 "--grid", $Grid, "--ensemble", "$Ensemble", "--workers", "$Workers",
                 "--de-budget", "$DeBudget", "--nm-iterations", "$NmIterations",
                 "--sigma-T", "$SigmaT", "--sigma-Ro", "$SigmaRo",
                 "--reg", "$Reg", "--seed", "$Seed",
                 "--out", (Join-Path $Out "inverse"))
if (-not $Uniform) {
    $InverseArgs += @("--adaptive-final", "--adaptive-base", $AdaptiveBase,
                      "--adaptive-level", "$AdaptiveLevel")
}
if ($NodeXs) { $InverseArgs += @("--node-xs", $NodeXs) }
if ($Epochs) { $InverseArgs += @("--epochs", $Epochs) }
if ($Twin) { $InverseArgs += @("--q-true", $Q) }
if ($QScalar -or -not $NoQScalar) { $InverseArgs += @("--q-scalar") }
if ($Theta0File) { $InverseArgs += @("--theta0-file", $Theta0File) }
& $JuliaExe @InverseArgs
if ($LASTEXITCODE -ne 0) { throw "run_inverse.jl failed" }

Write-Host "=== Stage 3/4: report ==="
& $JuliaExe "--project=$R3" "-t" "$Threads" (Join-Path $R3 "report_full.jl") `
    --direct $RunDir --inverse (Join-Path $Out "inverse") --out (Join-Path $Out "report.html")
if ($LASTEXITCODE -ne 0) { throw "report_full.jl failed" }

Write-Host "=== Stage 4/4: xlsx export ==="
& $JuliaExe "--project=$R3" "-t" "$Threads" (Join-Path $R3 "export_xlsx.jl") `
    --run $RunDir --data-dir $DataDir `
    --inverse (Join-Path $Out "inverse") --inv-data-dir $InvDataDir
if ($LASTEXITCODE -ne 0) { throw "export_xlsx.jl failed" }
Write-Host "=== summary ==="
Write-Host "report   : $(Join-Path $Out 'report.html')"
Write-Host "forward  : $RunDir"
Write-Host "inverse  : $(Join-Path $Out 'inverse')"
$Report = Join-Path $Out "report.html"
if ($OpenReport -and (Test-Path $Report)) { Start-Process $Report }
