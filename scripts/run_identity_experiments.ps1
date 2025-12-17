$Root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Root ".venv\Scripts\python.exe"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

$commonBase = @(
    "--demo",
    "--steps", "150",
    "--frame-every", "15",
    "--init-density", "0.15",
    "--perturb-on",
    "--perturb-t", "75",
    "--perturb-rate", "0.02",
    "--perturb-radius", "10"
)

$modes = @(
    @{ name = "object_off_tracked"; args = @("--observer-mode","off","--track-global-core") },
    @{ name = "passive_nopersist";  args = @("--observer-mode","passive","--no-subject-persist","--subject-radius","18","--subject-step","1","--subject-alpha","3.5","--subject-beta","3.0","--subject-core-min","40") },
    @{ name = "passive_persist";    args = @("--observer-mode","passive","--subject-radius","18","--subject-step","1","--subject-alpha","3.5","--subject-beta","3.0","--subject-core-min","40") },
    @{ name = "active_persist";     args = @("--observer-mode","active","--subject-radius","18","--subject-step","1","--subject-alpha","3.5","--subject-beta","3.0","--subject-core-min","40") }
)

$metricsPaths = @()

for ($seed = 0; $seed -le 0; $seed++) {
  foreach ($edge in @("fixed0","wrap")) {
    foreach ($m in $modes) {
        $tag = "$($m.name)_$edge`_seed$seed"
        $outDir = Join-Path "outputs" ("{0}_{1}" -f $ts, $tag)
        $common = $commonBase + @("--seed", "$seed", "--experiment-tag", $tag, "--out", $outDir, "--edge-mode", $edge)
        & $python "bda.py" @common @($m.args)
        if ($seed -eq 0) {
            $metricsPaths += (Join-Path $outDir "metrics.csv")
        }
    }
  }
}

# Compare the seed0 runs for quick plots
& $python "analysis\compare_runs.py" @metricsPaths "--out" "analysis_outputs" "--summary-csv" "analysis_outputs\identity_summary_seed0.csv"

# Aggregate across all seeds
$allMetrics = Get-ChildItem -Path (Join-Path "outputs" ($ts + "*")) -Recurse -Filter "metrics.csv" | Select-Object -ExpandProperty FullName
if ($allMetrics.Count -gt 0) {
    & $python "analysis\compare_runs.py" @allMetrics "--out" "analysis_outputs" "--summary-csv" "analysis_outputs\identity_summary_allseeds.csv"
}

