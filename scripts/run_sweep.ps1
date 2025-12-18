param(
  [string]$seeds = "0,1,2",
  [int]$steps = 300,
  [int]$perturbT = 150,
  [double]$perturbRate = 0.03,
  [int]$perturbRadius = 10,
  [string]$outRoot = "outputs"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$PY   = Join-Path $ROOT ".venv\Scripts\python.exe"
$BDA  = Join-Path $ROOT "bda.py"

if (!(Test-Path $PY))  { throw "Python venv not found: $PY" }
if (!(Test-Path $BDA)) { throw "Entrypoint not found: $BDA" }

# Parse seeds robustly (handles comma list or range a..b)
$seedList = @()
if ($seeds -match "^\s*\d+\s*\.\.\s*\d+\s*$") {
  $parts = $seeds -split "\.\."
  $start = [int]($parts[0].Trim())
  $end = [int]($parts[1].Trim())
  $seedList = $start..$end
} else {
  foreach ($s in ($seeds -split ",")) {
    $t = $s.Trim()
    $t = $t.Trim("'").Trim('"')
    if ($t -ne "") { $seedList += [int]$t }
  }
}
if ($seedList.Count -eq 0) { throw "No seeds parsed from: $seeds" }

New-Item -ItemType Directory -Force (Join-Path $ROOT "analysis_outputs") | Out-Null
$manifest = Join-Path $ROOT "analysis_outputs\sweep_manifest.txt"
Set-Content -Path $manifest -Value "" -Encoding UTF8

$modes = @(
  @{ name="object_off_tracked"; args=@("--observer-mode","off","--track-global-core") },
  @{ name="passive_nopersist";  args=@("--observer-mode","passive","--no-subject-persist") },
  @{ name="passive_persist";    args=@("--observer-mode","passive") },
  @{ name="active_persist";     args=@("--observer-mode","active") },
  @{ name="active_random";      args=@("--observer-mode","active_random") }
)

$boundaryModes = @("zero","wrap")

Write-Host "RUN_SWEEP: seeds=[$($seedList -join ',')], steps=$steps, boundaryModes=[$($boundaryModes -join ',')]" -ForegroundColor Cyan

foreach ($seed in $seedList) {
  foreach ($bmode in $boundaryModes) {
    foreach ($m in $modes) {

      $tag = "{0}_{1}_seed{2}" -f $m.name, $bmode, $seed

      $outDir = Join-Path $ROOT (Join-Path $outRoot $tag)
      New-Item -ItemType Directory -Force -Path $outDir | Out-Null

      $baseArgs = @(
        $BDA,
        "--demo",
        "--steps", "$steps",
        "--seed", "$seed",
        "--boundary-mode", "$bmode",
        "--experiment-tag", "$tag",
        "--run-name", "$tag",
        "--out", "$outDir",
        "--no-png",
        "--no-gif",
        "--frame-every", "999999",
        "--subject-radius", "18",
        "--subject-step", "1",
        "--subject-core-min", "40",
        "--subject-alpha", "3.5",
        "--subject-beta", "3.0",
        "--perturb-on",
        "--perturb-t", "$perturbT",
        "--perturb-rate", "$perturbRate",
        "--perturb-radius", "$perturbRadius"
      ) + $m.args

      Write-Host ("RUN: " + $tag) -ForegroundColor Yellow
      & $PY @baseArgs

      $metricsPath = Join-Path $outDir "metrics.csv"
      if (!(Test-Path $metricsPath)) { throw "metrics.csv missing for run: $outDir" }

      Add-Content -Path $manifest -Value $metricsPath -Encoding UTF8
    }
  }
}

Write-Host "DONE. Manifest written to: $manifest" -ForegroundColor Green

