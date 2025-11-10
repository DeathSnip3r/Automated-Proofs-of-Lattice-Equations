param(
  [string]$OutRoot = "results\rehearsal\extremes",
  [string]$CheckExe = "bin\check.exe",
  [string]$GenExe = "bin\lattice_gen.exe",
  [string[]]$Shapes = @('bestcase','worstcase'),
  [int[]]$Budgets = (1..14),
  [int]$Samples = 16,
  [int]$HighDepthThreshold = 13,
  [int]$HighDepthSamples = 2,
  [UInt64]$SeedBase = 6000000000,
  [string[]]$Engines = @('whitman','freese','cosma','hunt'),
  [int]$CosmaDepthCap = 13,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Ensure-Dir($p){ if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }

if (-not (Test-Path $GenExe)) { throw "Missing lattice generator at $GenExe" }
if (-not (Test-Path $CheckExe)) { throw "Missing check executable at $CheckExe" }

Ensure-Dir $OutRoot

$manifest = @()

foreach ($shape in $Shapes) {
  $shapeDir = Join-Path $OutRoot $shape
  Ensure-Dir $shapeDir

  $shapeEntries = @()
  foreach ($budget in $Budgets) {
    $samplesThisDepth = if ($budget -gt $HighDepthThreshold) { [math]::Max(1, $HighDepthSamples) } else { [math]::Max(1, $Samples) }
    $seed = $SeedBase + [UInt64]([Array]::IndexOf($Shapes,$shape) * 100000) + [UInt64]$budget
    $jsonPath = Join-Path $shapeDir ("pairs_B{0}.jsonl" -f $budget)

    Write-Host ("[gen] shape={0} B={1} seed={2}" -f $shape,$budget,$seed)
    if (-not $DryRun) {
      & $GenExe rand --seed $seed --vars 4096 --budget $budget --samples $samplesThisDepth `
        --shape $shape --min_arity 2 --max_arity 2 --p_join 0.5 --p_alt 1.0 --out $jsonPath
    }

    $entry = [ordered]@{
      shape = $shape
      budget = $budget
      seed = $seed
      samples = $samplesThisDepth
      json = [System.IO.Path]::GetFileName($jsonPath)
    }

    foreach ($engine in $Engines) {
      $csvPath = Join-Path $shapeDir ("{0}_B{1}.csv" -f $engine,$budget)

      # Skip Freese and Cosmadakis at worstcase depth 14 to avoid memory blow-ups
      if ($shape -eq 'worstcase' -and $budget -eq 14 -and ($engine -eq 'freese' -or $engine -eq 'cosma')) {
        Write-Host ("  [skip] {0} shape={1} depth={2} (excluded for safety)" -f $engine,$shape,$budget)
        if (-not $DryRun -and (Test-Path $csvPath)) { Remove-Item $csvPath -Force }
        $entry["csv_$engine"] = ""
        continue
      }

      # Cosmadakis general cap (already present) — skip above cap as well
      if ($engine -eq 'cosma' -and $budget -gt $CosmaDepthCap) {
        Write-Host ("  [skip] {0} shape={1} depth={2} (cap={3})" -f $engine,$shape,$budget,$CosmaDepthCap)
        if (-not $DryRun -and (Test-Path $csvPath)) { Remove-Item $csvPath -Force }
        $entry["csv_$engine"] = ""
        continue
      }

      Write-Host ("  [run] {0} shape={1} B={2}" -f $engine,$shape,$budget)
      if (-not $DryRun) {
        & $CheckExe --engine $engine --stats --json $jsonPath | Set-Content -Encoding ascii $csvPath
      }
      $entry["csv_$engine"] = [System.IO.Path]::GetFileName($csvPath)
    }

    $shapeEntries += [pscustomobject]$entry
  }

  if (-not $DryRun -and $shapeEntries.Count -gt 0) {
    $sumPath = Join-Path $shapeDir "manifest.csv"
    $shapeEntries | Sort-Object budget | Export-Csv -NoTypeInformation -Encoding ascii $sumPath
    $manifest += [pscustomobject]@{
      shape = $shape
      output_dir = $shapeDir
      budgets = ($Budgets -join ',')
      samples = $Samples
      manifest_csv = [System.IO.Path]::GetFileName($sumPath)
    }
  }
}

if (-not $DryRun -and $manifest.Count -gt 0) {
  $rootManifest = Join-Path $OutRoot "manifest.csv"
  $manifest | Export-Csv -NoTypeInformation -Encoding ascii $rootManifest
  Write-Host ("[ok] extreme manifest -> {0}" -f $rootManifest)
}
