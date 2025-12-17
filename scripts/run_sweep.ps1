param(
    [string]$seeds = "0,1,2,3,4",
    [int]$steps = 300,
    [int]$perturbT = 150,
    [string]$boundaryModes = "zero,wrap",
    [string]$outRoot = "outputs"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Root ".venv\Scripts\python.exe"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

$seedList = $seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$boundaryList = $boundaryModes.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

$modes = @(
    @{ name = "off_tracked"; args = @("--observer-mode","off","--track-global-core") },
    @{ name = "passive_nopersist"; args = @("--observer-mode","passive","--no-subject-persist","--subject-radius","18","--subject-step","1","--subject-core-min","40","--subject-alpha","3.5","--subject-beta","3.0") },
    @{ name = "passive_persist"; args = @("--observer-mode","passive","--subject-radius","18","--subject-step","1","--subject-core-min","40","--subject-alpha","3.5","--subject-beta","3.0") },
    @{ name = "active_persist"; args = @("--observer-mode","active","--subject-radius","18","--subject-step","1","--subject-core-min","40","--subject-alpha","3.5","--subject-beta","3.0") }
)

$common = @(
    "--H","180","--W","240",
    "--steps",$steps.ToString(),
    "--frame-every","1",
    "--no-png",
    "--no-gif",
    "--perturb-on",
    "--perturb-t",$perturbT.ToString(),
    "--perturb-rate","0.03",
    "--perturb-radius","10"
)

$manifestPath = Join-Path $Root "analysis_outputs\sweep_manifest.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $manifestPath) | Out-Null

if (Test-Path $manifestPath) { Remove-Item $manifestPath -Force }
"tag,boundary_mode,observer_mode,seed,metrics_path" | Out-File -FilePath $manifestPath -Encoding UTF8

foreach ($seed in $seedList) {
  foreach ($bmode in $boundaryList) {
    foreach ($m in $modes) {
      $tag = "seed${seed}_${bmode}_${m.name}"
      $outDir = Join-Path $outRoot ("{0}_{1}" -f $ts, $tag)
      $argsAll = $common + @("--seed",$seed,"--boundary-mode",$bmode,"--experiment-tag",$tag,"--out",$outDir) + $m.args
      Write-Host "RUN seed=$seed boundary=$bmode mode=$($m.name) -> $outDir"
      & $python "bda.py" @argsAll
      $metricsPath = Join-Path $outDir "metrics.csv"
      "$tag,$bmode,$($m.name),$seed,$metricsPath" | Out-File -FilePath $manifestPath -Append -Encoding UTF8
    }
  }
}

Write-Host "DONE sweep. Manifest: $manifestPath"

