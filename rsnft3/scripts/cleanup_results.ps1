<#
.SYNOPSIS
    6.4: дедуп и разбор results/ — отчёт об объёме, удаление только по явному подтверждению.

.EXAMPLE
    .\cleanup_results.ps1                          # только отчёт об объёме
    .\cleanup_results.ps1 -DeleteDupCache          # удалить дубли _input_cache в прогонах
    .\cleanup_results.ps1 -DeleteBench             # удалить benchmark/тестовые каталоги
    .\cleanup_results.ps1 -DeleteOld 30            # удалить прогоны старше 30 дней
#>
[CmdletBinding()]
param(
    [switch]$DeleteDupCache,
    [switch]$DeleteBench,
    [int]   $DeleteOld = -1
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null
} catch { }

$Results = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "..\results"
$Results = [System.IO.Path]::GetFullPath($Results)
if (-not (Test-Path $Results)) { Write-Host "нет каталога $Results"; exit 0 }

Write-Host "=== объём results/ ==="
"{0:N1} МБ" -f ((Get-ChildItem $Results -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)

Write-Host ""
Write-Host "=== дубли _input_cache внутри прогонов (общий теперь results/_input_cache) ==="
$Dups = Get-ChildItem $Results -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -Directory -Filter "_input_cache" -ErrorAction SilentlyContinue
}
$DupSize = ($Dups | ForEach-Object {
    (Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum
} | Measure-Object -Sum).Sum
Write-Host ("итого: {0} дублей, {1:N1} МБ" -f @($Dups).Count, ($DupSize / 1MB))
if ($DeleteDupCache) {
    $Dups | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
    Write-Host "удалено"
} else {
    Write-Host "(удаление: -DeleteDupCache)"
}

Write-Host ""
Write-Host "=== временные benchmark-каталоги (_bench*, _ageprobe, _test*) ==="
$Bench = Get-ChildItem $Results -Directory | Where-Object { $_.Name -match '^(_bench|_ageprobe|_test)' }
if ($Bench) {
    foreach ($b in $Bench) {
        $s = ((Get-ChildItem $b.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
        Write-Host ("{0}  {1:N1} МБ" -f $b.FullName, $s)
    }
    if ($DeleteBench) { $Bench | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }; Write-Host "удалено" }
    else { Write-Host "(удаление: -DeleteBench)" }
} else { Write-Host "(нет)" }

if ($DeleteOld -ge 0) {
    Write-Host ""
    Write-Host "=== прогоны старше $DeleteOld дн. ==="
    $Cut = (Get-Date).AddDays(-$DeleteOld)
    $Old = Get-ChildItem $Results -Directory | Where-Object { $_.LastWriteTime -lt $Cut -and $_.Name -notlike "_*" }
    if ($Old) {
        foreach ($o in $Old) {
            $s = ((Get-ChildItem $o.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
            Write-Host ("{0}  {1:N1} МБ" -f $o.FullName, $s)
        }
        Write-Host "УДАЛЯЕМ (финальное подтверждение -DeleteOld):"
        $Old | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
    } else { Write-Host "(нет)" }
}

Write-Host ""
Write-Host "готово (без -Delete* ничего не удалено)"
