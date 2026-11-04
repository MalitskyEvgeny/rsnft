<#
.SYNOPSIS
    Adaptive-grid inverse run with report, mesh pictures and a final summary.

.DESCRIPTION
    Wraps rsnft3/run_inverse.jl (adaptive backend) + rsnft3/report_inverse.jl,
    then prints the key acceptance numbers from solution.json and the
    paths of the generated report/mesh artifacts.

    Modes:
      final  (-Mode final,  default)  search on the regular -Grid, then the
              adaptive control stage at the best theta (--adaptive-final).
      search (-Mode search)           the ENTIRE pipeline on the adaptive grid
              (--adaptive-search); no uniform grid anywhere.

.EXAMPLE
    .\run_adaptive.ps1                                        # quick control run
    .\run_adaptive.ps1 -Mode search -AdaptiveBase 16,32 -AdaptiveLevel 1
    .\run_adaptive.ps1 -Grid 30,60 -AdaptiveBase 30,60 -AdaptiveLevel 1 -OpenReport
#>
[CmdletBinding()]
param(
    [string]$RepoRoot      = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$JuliaExe      = "julia",
    [string]$DataDir       = "",
    [string]$Wells         = "",
    [ValidateSet("final","search")]
    [string]$Mode          = "final",
    [string]$OutDir        = "",
    [string]$Grid          = "20,40",
    [string]$AdaptiveBase  = "16,32",
    [int]   $AdaptiveLevel = 2,
    [string]$AdaptivePreview = "120,160",
    [string]$Theta0File    = "",
    [int]   $Threads       = 8,
    [int]   $Ensemble      = 1,
    [int]   $Workers       = 1,
    [int]   $DeBudget      = 2000,
    [int]   $NmIterations  = 30,
    [double]$Reg           = 1.0,
    [int]   $Seed          = 1,
    [double]$SigmaT        = 2.0,
    [double]$SigmaRo       = 0.05,
    [double]$MaxDtMa       = 2.5,
    [string]$PcgTol        = "1e-4",
    [switch]$FrozenMesh,
    [switch]$NoReport,
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"

$Inv  = [System.Globalization.CultureInfo]::InvariantCulture
$R3   = Join-Path $RepoRoot "rsnft3"
$Proj = Join-Path $R3 "Project.toml"

if (-not (Test-Path $Proj))            { throw "rsnft3 project not found under $RepoRoot" }
if (-not (Get-Command $JuliaExe -ErrorAction SilentlyContinue)) { throw "julia not found on PATH" }
if (-not $DataDir) {
    $DataDir = Join-Path $RepoRoot ([char]0x0414 + [string][char]0x0430 + [char]0x043D + [char]0x043D + [char]0x044B + [char]0x0435 + " " + [char]0x0434 + [char]0x043B + [char]0x044F + " " + [char]0x043E + [char]0x0431 + [char]0x0440 + [char]0x0430 + [char]0x0442 + [char]0x043D + [char]0x043E + [char]0x0439 + " " + [char]0x0437 + [char]0x0430 + [char]0x0434 + [char]0x0430 + [char]0x0447 + [char]0x0438)
}
if (-not (Test-Path $DataDir))         { throw "data dir not found: $DataDir" }
if ($Theta0File -and -not (Test-Path $Theta0File)) { throw "theta0-file not found: $Theta0File" }

if (-not $Wells) {
    $Wells = (Join-Path $DataDir "TemperatureWellData.xlsx") + ";" +
             (Join-Path $DataDir "VitriniteWellData.xlsx")
}
if (-not $OutDir) {
    $Stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = Join-Path $RepoRoot ("results/adaptive_{0}_{1}" -f $Mode, $Stamp)
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$AdaptiveFlag = @{ }
if     ($Mode -eq "final")  { $AdaptiveFlag["--adaptive-final"]  = $null }
elseif ($Mode -eq "search") { $AdaptiveFlag["--adaptive-search"] = $null }
if ($FrozenMesh)            { $AdaptiveFlag["--adaptive-frozen"] = $null }

$InvArgs = @(
    "--data-dir", $DataDir,
    "--wells", $Wells,
    "--grid", $Grid,
    "--ensemble", $Ensemble, "--workers", $Workers,
    "--de-budget", $DeBudget, "--nm-iterations", $NmIterations,
    "--sigma-T", $SigmaT.ToString($Inv), "--sigma-Ro", $SigmaRo.ToString($Inv),
    "--reg", $Reg.ToString($Inv), "--seed", $Seed,
    "--max-dt-ma", $MaxDtMa.ToString($Inv), "--pcg-tol", $PcgTol,
    "--adaptive-base", $AdaptiveBase, "--adaptive-level", $AdaptiveLevel,
    "--adaptive-preview", $AdaptivePreview,
    "--out", $OutDir
) + @($AdaptiveFlag.Keys)
if ($Theta0File) { $InvArgs += @("--theta0-file", (Resolve-Path $Theta0File).Path) }

Write-Host "=== run_inverse ($Mode) ===" -ForegroundColor Cyan
Write-Host ("out={0}  grid={1}  adaptive base={2} level={3} preview={4}" -f $OutDir, $Grid, $AdaptiveBase, $AdaptiveLevel, $AdaptivePreview)
$T0 = [System.Diagnostics.Stopwatch]::StartNew()

& $JuliaExe "--project=$R3" "-t" $Threads (Join-Path $R3 "run_inverse.jl") @InvArgs
if ($LASTEXITCODE -ne 0) { throw "run_inverse.jl failed with exit code $LASTEXITCODE" }
Write-Host ("inverse done in {0:mm\:ss}" -f $T0.Elapsed) -ForegroundColor Green

$SolutionPath = Join-Path $OutDir "solution.json"
if (-not (Test-Path $SolutionPath)) { throw "solution.json missing: $SolutionPath" }

if (-not $NoReport) {
    Write-Host "`n=== report ===" -ForegroundColor Cyan
    & $JuliaExe "--project=$R3" (Join-Path $R3 "report_inverse.jl") $OutDir
    if ($LASTEXITCODE -ne 0) { Write-Warning "report_inverse.jl failed; continuing" }
}

$sol = Get-Content $SolutionPath -Raw -Encoding UTF8 | ConvertFrom-Json
$best, $cr = $sol.best, $sol.criteria
Write-Host "`n=== summary ===" -ForegroundColor Cyan
Write-Host ("status   : {0}" -f $sol.status)
Write-Host ("best J   : {0}  (J_T={1}, J_Ro={2}, J_reg={3})" -f `
    $best.J.ToString($Inv), $best.J_T.ToString($Inv), $best.J_Ro.ToString($Inv), $best.J_reg.ToString($Inv))
Write-Host ("rmse_T   : {0}   (accept: <= 4 = 2*sigma_T)" -f $cr.rmse_T.ToString($Inv))
Write-Host ("rmse_Ro  : {0}   (accept: <= 0.10)" -f $cr.rmse_Ro.ToString($Inv))
Write-Host ("T in 2s  : {0}%  (accept: >= 95%)" -f (100 * $cr.frac_2s_T).ToString($Inv))
Write-Host ("Ro in 2s : {0}%" -f (100 * $cr.frac_2s_Ro).ToString($Inv))
if ($sol.adaptive) {
    $acr = $sol.adaptive.criteria
    Write-Host ("adaptive : base={0} level={1}  rmse_T={2}  rmse_Ro={3}  T in 2s={4}%" -f `
        $sol.adaptive.base, $sol.adaptive.level,
        $acr.rmse_T.ToString($Inv), $acr.rmse_Ro.ToString($Inv), (100 * $acr.frac_2s_T).ToString($Inv)) -ForegroundColor Yellow
}
Write-Host ("theta    : {0}" -f (($best.theta | ForEach-Object { $_.ToString("F2", $Inv) }) -join ", "))

$Mesh = Join-Path $OutDir "mesh_adaptive.png"
if (Test-Path $Mesh) { Write-Host ("mesh     : {0}" -f $Mesh) -ForegroundColor Yellow }
$Report = Join-Path $OutDir "report.html"
if (Test-Path $Report) { Write-Host ("report   : {0}" -f $Report) }

if ($OpenReport -and (Test-Path $Report)) { Start-Process $Report }
Write-Host ("total    : {0:hh\:mm\:ss}" -f $T0.Elapsed) -ForegroundColor Green
