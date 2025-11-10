param(
  [UInt64]$SeedBase = 8589934592,
  [ValidateSet('auto','skip')][string]$Build = 'auto',
  [string]$OutRoot = "results\stress",
  [string[]]$Filter = @(),
  [Nullable[int]]$SampleOverride,
  [int]$BudgetCeiling = 0,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$BinDir   = "bin"
$CheckExe = Join-Path $BinDir "check.exe"
$GenExe   = Join-Path $BinDir "lattice_gen.exe"
$engines  = @('whitman','freese','cosma','hunt')

function Ensure-Dir($p){ if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }
function Try-Make {
  foreach ($mk in @('make','mingw32-make')) {
    $cmd = Get-Command $mk -ErrorAction SilentlyContinue
    if ($cmd) { & $mk all; return $true }
  }
  return $false
}
function Gpp-Or-DIE { if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) { throw "g++ not found" } }

function Quantiles([double[]]$xs){
  if (-not $xs -or $xs.Count -eq 0) { return @{p25=0.0; p50=0.0; p75=0.0} }
  $ys = $xs | Sort-Object
  $n = $ys.Count
  return @{
    p25 = $ys[[math]::Floor(0.25*($n-1))]
    p50 = $ys[[math]::Floor(0.50*($n-1))]
    p75 = $ys[[math]::Floor(0.75*($n-1))]
  }
}
function Mean([double[]]$xs){
  if (-not $xs -or $xs.Count -eq 0) { return 0.0 }
  return ($xs | Measure-Object -Average).Average
}

function Parse-StatsJson([string]$cell){
  if ([string]::IsNullOrWhiteSpace($cell)) { return $null }
  $clean = $cell.Trim()
  if ($clean.StartsWith('"') -and $clean.EndsWith('"')) {
    $clean = $clean.Substring(1, $clean.Length - 2)
  }
  $clean = $clean -replace '""','"'
  if ([string]::IsNullOrWhiteSpace($clean)) { return $null }
  try {
    return $clean | ConvertFrom-Json
  } catch {
    Write-Warning "Failed to parse stats JSON: $clean"
    return $null
  }
}

function New-StatsObject($experiment,$shape,$budget,$engine,$pairId,$direction,$statsObj){
  if (-not $statsObj) { return $null }
  $base = [ordered]@{
    experiment = $experiment
    shape      = $shape
    budget     = $budget
    engine     = $engine
    pair_id    = [int]$pairId
    direction  = $direction
  }
  foreach ($prop in $statsObj.PSObject.Properties){
    $base[$prop.Name] = $prop.Value
  }
  return [pscustomobject]$base
}

$experiments = New-Object System.Collections.ArrayList

$experiments.Add([pscustomobject]@{
  Id        = 'H1_balanced_noise'
  Mode      = 'standard'
  Shape     = 'balanced'
  ShapeLabel= 'balanced'
  Budgets   = @(200)
  Vars      = 8192
  Samples   = 256
  MinArity  = 2
  MaxArity  = 6
  PJoin     = 0.5
  PAlt      = 0.8
  EnsureUniqueLeaves = $false
  URoot     = 'auto'
  VRoot     = 'auto'
}) | Out-Null

$pJoins = @(0.3,0.5,0.7)
$pAlts  = @(0.2,0.5,0.8)
foreach ($pj in $pJoins) {
  foreach ($pa in $pAlts) {
    $pjTag = [int]([math]::Round($pj * 100))
    $paTag = [int]([math]::Round($pa * 100))
    $experiments.Add([pscustomobject]@{
      Id        = "H2_rand_pj${pjTag}_pa${paTag}"
      Mode      = 'standard'
      Shape     = 'balanced'
      ShapeLabel= "rand"
      Budgets   = @(50)
      Vars      = 4096
      Samples   = 256
      MinArity  = 2
      MaxArity  = 6
      PJoin     = [double]$pj
      PAlt      = [double]$pa
      EnsureUniqueLeaves = $false
      URoot     = 'auto'
      VRoot     = 'auto'
    }) | Out-Null
  }
}

$experiments.Add([pscustomobject]@{
  Id        = 'H3_hybrid_bag'
  Mode      = 'bag'
  ShapeLabel= 'bag'
  Shape     = 'hybrid'
  BagShapes = @('balanced','alternating','leftspine')
  BagWeights= @()
  Budgets   = @(100)
  Vars      = 4096
  Samples   = 256
  MinArity  = 2
  MaxArity  = 6
  PJoin     = 0.5
  PAlt      = 0.6
  URoot     = 'auto'
  VRoot     = 'auto'
  EnsureUniqueLeaves = $false
}) | Out-Null

$experiments = [array]$experiments

if ($Filter -and $Filter.Count -gt 0) {
  $experiments = $experiments | Where-Object { $Filter -contains $_.Id }
  if ($experiments.Count -eq 0) {
    Write-Warning "Filter matched no experiments. Exiting."
    return
  }
}

Ensure-Dir $OutRoot
Ensure-Dir $BinDir

if ($Build -ne 'skip') {
  $ok = $false
  try { $ok = Try-Make } catch { $ok = $false }
  if (-not (Test-Path $CheckExe) -or -not (Test-Path $GenExe)) {
    Gpp-Or-DIE
    if (-not $DryRun) {
      Write-Host "[build] compiling directly..."
      & g++ -O2 -std=c++17 -I include `
        src/whitman.cpp src/freese.cpp src/cosmadakis.cpp src/hunt.cpp src/runner_min.cpp `
        -o $CheckExe
      & g++ -O2 -std=c++17 -I include src/lattice_gen.cpp -o $GenExe
    }
  }
}

if (-not (Test-Path $CheckExe)) { throw "Missing $CheckExe" }
if (-not (Test-Path $GenExe))   { throw "Missing $GenExe" }

$manifest = @()
$expIndex = 0

foreach ($exp in $experiments) {
  $expIndex += 1
  $expDir = Join-Path $OutRoot $exp.Id
  Ensure-Dir $expDir

  $summary = @()
  $metaRows = @()
  $statsRows = @()
  if ($SampleOverride.HasValue) {
    $samplesUsed = [int]$SampleOverride.Value
  } else {
    $samplesUsed = [int]$exp.Samples
  }

  $budgets = $exp.Budgets
  if ($BudgetCeiling -gt 0) {
    $budgets = $budgets | Where-Object { $_ -le $BudgetCeiling }
  }
  if (-not $budgets -or $budgets.Count -eq 0) {
    Write-Host ("[skip] {0} (no budgets ≤ {1})" -f $exp.Id,$BudgetCeiling)
    continue
  }

  foreach ($B in $budgets) {
    $seed = $SeedBase + ($expIndex * 10000) + $B
    $vars = [int]$exp.Vars

    $jsonPath = Join-Path $expDir ("pairs_B{0}.jsonl" -f $B)

    if ($exp.Mode -eq 'bag') {
      $shapes = $exp.BagShapes
      if (-not $shapes -or $shapes.Count -eq 0) {
        throw "Experiment $($exp.Id) has Mode=bag but no BagShapes defined"
      }
      $weights = if ($exp.PSObject.Properties.Name -contains 'BagWeights' -and $exp.BagWeights.Count -eq $shapes.Count -and ($exp.BagWeights | Measure-Object -Sum).Sum -gt 0) { $exp.BagWeights } else { @(for($i=0;$i -lt $shapes.Count;$i++){1}) }
      $weightSum = ($weights | Measure-Object -Sum).Sum
      $alloc = New-Object System.Collections.Generic.List[int]
      $assigned = 0
      for ($i=0; $i -lt $shapes.Count; $i++) {
        $portion = [math]::Floor($samplesUsed * $weights[$i] / $weightSum)
        $alloc.Add([int]$portion)
        $assigned += $portion
      }
      $remainder = $samplesUsed - $assigned
      $idx = 0
      while ($remainder -gt 0 -and $shapes.Count -gt 0) {
        $alloc[$idx % $shapes.Count]++
        $remainder--
        $idx++
      }

      Write-Host ("[gen] {0} B={1} seed={2} (bag)" -f $exp.Id,$B,$seed)

      if (-not $DryRun) {
        $bagFiles = @()
        for ($i=0; $i -lt $shapes.Count; $i++) {
          $shape = $shapes[$i]
          $count = $alloc[$i]
          if ($count -le 0) { continue }
          $shapeSeed = $seed + (($i+1) * 1000)
          $tmpPath = [System.IO.Path]::GetTempFileName()
          $genArgs = @(
            'rand','--seed',$shapeSeed,'--vars',$vars,'--budget',$B,'--samples',$count,
            '--shape',$shape,'--min_arity',$exp.MinArity,'--max_arity',$exp.MaxArity,
            '--p_join',$exp.PJoin,'--p_alt',$exp.PAlt,'--out',$tmpPath
          )
          if ($shape -eq 'alternating' -and ($exp.URoot -ne 'auto' -or $exp.VRoot -ne 'auto')) {
            $genArgs += @('--u_root',$exp.URoot,'--v_root',$exp.VRoot)
          }
          & $GenExe @genArgs
          $bagFiles += $tmpPath
        }
        $allLines = New-Object System.Collections.Generic.List[string]
        foreach ($tmp in $bagFiles) {
          $lines = Get-Content $tmp
          foreach ($ln in $lines) { $allLines.Add($ln) }
          Remove-Item $tmp -Force
        }
        $allLines | Set-Content -Encoding ascii $jsonPath
      }
    } else {
      $genArgs = @(
        'rand','--seed',$seed,'--vars',$vars,'--budget',$B,'--samples',$samplesUsed,
        '--shape',$exp.Shape,'--min_arity',$exp.MinArity,'--max_arity',$exp.MaxArity,
        '--p_join',$exp.PJoin,'--p_alt',$exp.PAlt,'--out',$jsonPath
      )
      if ($exp.Shape -eq 'alternating' -and ($exp.URoot -ne 'auto' -or $exp.VRoot -ne 'auto')) {
        $genArgs += @('--u_root',$exp.URoot,'--v_root',$exp.VRoot)
      }
      Write-Host ("[gen] {0} B={1} seed={2}" -f $exp.Id,$B,$seed)
      if (-not $DryRun) { & $GenExe @genArgs }
    }

    if ($DryRun) {
      foreach ($e in $engines) {
        Write-Host ("  [run] {0} B={1}" -f $e,$B)
      }
      continue
    }

    $lineObjs = Get-Content $jsonPath | ForEach-Object { $_ | ConvertFrom-Json }
    $pairDepths = @()
    $pairAlt = @()
    $pairShare = @()
    foreach ($line in $lineObjs) {
      if ($line.meta) {
        $pairDepths += [double][math]::Max($line.meta.u.height, $line.meta.v.height)
        $pairAlt += [double]([double]$line.meta.pair.avg_alt_index)
        $pairShare += [double]([double]$line.meta.pair.avg_share_ratio)
        $shapeTag = if ($exp.Mode -eq 'bag' -and $line.config.shape) { $line.config.shape } else { $exp.ShapeLabel }
        $metaRows += [pscustomobject]@{
          experiment = $exp.Id
          budget = $B
          shape_tag = $shapeTag
          u_nodes = [int]$line.meta.u.nodes
          v_nodes = [int]$line.meta.v.nodes
          u_height = [int]$line.meta.u.height
          v_height = [int]$line.meta.v.height
          avg_alt_index = [double]$line.meta.pair.avg_alt_index
          avg_share_ratio = [double]$line.meta.pair.avg_share_ratio
        }
      } else {
        $pairDepths += [double]$B
      }
    }
    $depthQ = Quantiles([double[]]$pairDepths)
    $altQ = Quantiles([double[]]$pairAlt)
    $shareQ = Quantiles([double[]]$pairShare)

    $engineCsvs = @{}
    foreach ($e in $engines) {
      $csvPath = Join-Path $expDir ("{0}_B{1}.csv" -f $e,$B)
      Write-Host ("  [run] {0} B={1}" -f $e,$B)
      & $CheckExe --engine $e --stats --json $jsonPath | Set-Content -Encoding ascii $csvPath
      $engineCsvs[$e] = $csvPath
    }

    foreach ($e in $engines) {
      $rows = Import-Csv $engineCsvs[$e]
      $totals = $rows | ForEach-Object { [double]$_.total_us }
      $q = Quantiles([double[]]$totals)
      $avg = Mean([double[]]$totals)
      foreach ($row in $rows) {
        $uvStats = Parse-StatsJson($row.uv_stats)
        $vuStats = Parse-StatsJson($row.vu_stats)
        $uvObj = New-StatsObject($exp.Id,$exp.ShapeLabel,$B,$e,$row.pair_id,'uv',$uvStats)
        if ($uvObj) { $statsRows += $uvObj }
        $vuObj = New-StatsObject($exp.Id,$exp.ShapeLabel,$B,$e,$row.pair_id,'vu',$vuStats)
        if ($vuObj) { $statsRows += $vuObj }
      }
      $summary += [pscustomobject]@{
        experiment        = $exp.Id
        shape             = $exp.ShapeLabel
        budget            = $B
        median_pairDepth  = [double]$depthQ.p50
        engine            = $e
        p25_us            = [double]$q.p25
        median_us         = [double]$q.p50
        p75_us            = [double]$q.p75
        mean_us           = [double][math]::Round($avg,3)
        n_pairs           = $rows.Count
        median_alt_index  = [double][math]::Round($altQ.p50,6)
        median_share      = [double][math]::Round($shareQ.p50,6)
        csv_path          = [System.IO.Path]::GetFileName($engineCsvs[$e])
        json_path         = [System.IO.Path]::GetFileName($jsonPath)
        p_join            = [double]$exp.PJoin
        p_alt             = [double]$exp.PAlt
      }
    }
  }

  if ($DryRun) { continue }

  $sumPath = Join-Path $expDir "summary.csv"
  $summary | Sort-Object engine,budget | Export-Csv -NoTypeInformation -Encoding ascii $sumPath
  $metaPath = $null
  if ($metaRows.Count -gt 0) {
    $metaPath = Join-Path $expDir "meta_pairs.csv"
    $metaRows | Export-Csv -NoTypeInformation -Encoding ascii $metaPath
  }

  $statsPath = $null
  $statsAggPath = $null
  if ($statsRows.Count -gt 0) {
    $statsPath = Join-Path $expDir "solver_stats.csv"
    $statsRows | Export-Csv -NoTypeInformation -Encoding ascii $statsPath

    $baseFields = @('experiment','shape','budget','engine','pair_id','direction')
    $metricHash = New-Object System.Collections.Generic.HashSet[string]
    foreach ($row in $statsRows) {
      foreach ($prop in $row.PSObject.Properties) {
        if (-not ($baseFields -contains $prop.Name)) {
          [void]$metricHash.Add($prop.Name)
        }
      }
    }
    $metricNames = $metricHash.ToArray() | Sort-Object

    $statAgg = @()
    $statsRows | Group-Object experiment,shape,budget,engine,direction | ForEach-Object {
      $first = $_.Group[0]
      $entry = [ordered]@{
        experiment = $first.experiment
        shape      = $first.shape
        budget     = $first.budget
        engine     = $first.engine
        direction  = $first.direction
        n          = $_.Count
      }
      foreach ($name in $metricNames) {
        $vals = @()
        foreach ($row in $_.Group) {
          if ($row.PSObject.Properties.Name -contains $name) {
            $value = $row.$name
            if ($null -ne $value -and $value -ne '') {
              try { $vals += [double]$value } catch { }
            }
          }
        }
        $entry["avg_$name"] = if ($vals.Count -gt 0) { ($vals | Measure-Object -Average).Average } else { 0 }
      }
      $statAgg += [pscustomobject]$entry
    }
    if ($statAgg.Count -gt 0) {
      $statsAggPath = Join-Path $expDir "solver_stats_agg.csv"
      $statAgg | Export-Csv -NoTypeInformation -Encoding ascii $statsAggPath
    }
  }

  $manifestEntry = [ordered]@{
    id = $exp.Id
    mode = $exp.Mode
    shape = $exp.ShapeLabel
    budgets = ($budgets -join ',')
    samples_requested = $exp.Samples
    samples_used = $samplesUsed
    vars = $exp.Vars
    base_seed = $SeedBase
    ensure_unique_leaves = $exp.EnsureUniqueLeaves
    output_dir = $expDir
    summary_csv = [System.IO.Path]::GetFileName($sumPath)
    meta_csv    = if ($metaPath) { [System.IO.Path]::GetFileName($metaPath) } else { '' }
    stats_csv   = if ($statsPath) { [System.IO.Path]::GetFileName($statsPath) } else { '' }
    stats_agg_csv = if ($statsAggPath) { [System.IO.Path]::GetFileName($statsAggPath) } else { '' }
    p_join = [double]$exp.PJoin
    p_alt  = [double]$exp.PAlt
  }
  if ($exp.Mode -eq 'bag') {
    $manifestEntry['bag_shapes'] = ($exp.BagShapes -join ',')
  }
  $manifest += [pscustomobject]$manifestEntry
}

if ($DryRun) { Write-Host "[dry-run] no files written."; return }

if ($manifest.Count -gt 0) {
  $manifestPath = Join-Path $OutRoot "manifest.csv"
  $manifest | Export-Csv -NoTypeInformation -Encoding ascii $manifestPath
  Write-Host ("[ok] manifest -> {0}" -f $manifestPath)
}
