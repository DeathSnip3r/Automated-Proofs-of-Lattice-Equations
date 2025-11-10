param(
  [UInt64]$SeedBase = 4294967296,
  [ValidateSet('auto','skip')][string]$Build = 'auto',
  [string]$OutRoot = "results\controlled",
  [string[]]$Filter = @(),
  [Nullable[int]]$SampleOverride,
  [int]$BudgetCeiling = 0,
  [int]$AltfullBudgetCeiling = 0,
  [switch]$DryRun,
  [ValidateSet('canonical','legacy','both')][string]$Representation = 'canonical'
)

$ErrorActionPreference = 'Stop'

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Gpp-Or-DIE {
  if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    throw 'g++ not found'
  }
}

function Try-Make {
  foreach ($mk in @('make','mingw32-make')) {
    $cmd = Get-Command $mk -ErrorAction SilentlyContinue
    if ($cmd) {
      try {
        & $mk all
        return $true
      } catch {
        Write-Warning "Build tool '$mk' failed: $($_.Exception.Message)"
      }
    }
  }
  return $false
}

function Quantiles {
  param([double[]]$xs)
  if (-not $xs -or $xs.Count -eq 0) {
    return @{p25 = 0.0; p50 = 0.0; p75 = 0.0}
  }
  $ys = $xs | Sort-Object
  $n = $ys.Count
  return @{
    p25 = $ys[[math]::Floor(0.25 * ($n - 1))]
    p50 = $ys[[math]::Floor(0.50 * ($n - 1))]
    p75 = $ys[[math]::Floor(0.75 * ($n - 1))]
  }
}

function Mean {
  param([double[]]$xs)
  if (-not $xs -or $xs.Count -eq 0) {
    return 0.0
  }
  return ($xs | Measure-Object -Average).Average
}

function Parse-StatsJson {
  param([string]$Cell)
  if ([string]::IsNullOrWhiteSpace($Cell)) { return $null }
  $clean = $Cell.Trim()
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

function New-StatsObject {
  param(
    [string]$Experiment,
    [string]$Shape,
    [int]$Budget,
    [string]$Engine,
    [int]$PairId,
    [string]$Direction,
    $StatsObj
  )
  if (-not $StatsObj) { return $null }
  $base = [ordered]@{
    experiment = $Experiment
    shape      = $Shape
    budget     = $Budget
    engine     = $Engine
    pair_id    = [int]$PairId
    direction  = $Direction
  }
  foreach ($prop in $StatsObj.PSObject.Properties) {
    $base[$prop.Name] = $prop.Value
  }
  return [pscustomobject]$base
}

$BinDir   = 'bin'
$CheckExe = Join-Path $BinDir 'check.exe'
$GenExe   = Join-Path $BinDir 'lattice_gen.exe'
$engines  = @('whitman','freese','cosma','hunt')
$isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
$linkArgs = if ($isWindows) { @('-lpsapi') } else { @() }

$experiments = @(
  [pscustomobject]@{
    Id = 'C1_altfull'
    Shape = 'altfull'
    Budgets = @(1..14)
    Vars = 4096
    Samples = 16
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $true
    SeedBase = 4294977296
    HighDepthStart = 12
    HighDepthSampleCap = 2
  },
  [pscustomobject]@{
    Id = 'C2_altfixed'
    Shape = 'alternating'
    Budgets = @(50,100,200)
    Vars = 2048
    Samples = 4
    MinArity = 2
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'meet'
    VRoot = 'join'
    EnsureUniqueLeaves = $false
    SeedBase = 4294987296
  },
  [pscustomobject]@{
    Id = 'C3_altpolarity'
    Shape = 'alternating'
    Budgets = @(50,100,200)
    Vars = 2048
    Samples = 4
    MinArity = 2
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'join'
    VRoot = 'meet'
    EnsureUniqueLeaves = $false
    SeedBase = 4294997296
  },
  [pscustomobject]@{
    Id = 'C4_share_low'
    Shape = 'balanced'
    Budgets = @(200,400,600)
    Vars = 64
    Samples = 4
    MinArity = 2
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 0.2
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295007296
    SeedOverrides = @{ 200 = 8000; 400 = 8300; 600 = 8350; 800 = 8400; 1000 = 8450 }
  },
  [pscustomobject]@{
    Id = 'C4_share_medium'
    Shape = 'balanced'
    Budgets = @(200,400,600)
    Vars = 256
    Samples = 4
    MinArity = 2
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 0.6
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295008296
    SeedOverrides = @{ 200 = 8900; 400 = 9001; 600 = 9051; 800 = 9101; 1000 = 9151 }
  },
  [pscustomobject]@{
    Id = 'C4_share_high'
    Shape = 'balanced'
    Budgets = @(200,400,600)
    Vars = 4096
    Samples = 4
    MinArity = 2
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 0.8
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295009296
    SeedOverrides = @{ 200 = 7019; 400 = 8700; 600 = 8750; 800 = 8800; 1000 = 8850 }
  },
  [pscustomobject]@{
    Id = 'C5_spines'
    Shape = 'leftspine'
    Budgets = @(20,40)
    Vars = 1024
    Samples = 4
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295017296
  },
  [pscustomobject]@{
    Id = 'C5_spines_r'
    Shape = 'rightspine'
    Budgets = @(20,40)
    Vars = 1024
    Samples = 4
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295027296
  },
  [pscustomobject]@{
    Id = 'C6_k2'
    Shape = 'alternating'
    Budgets = @(40,80,160)
    Vars = 4096
    Samples = 4
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'meet'
    VRoot = 'join'
    EnsureUniqueLeaves = $false
    SeedBase = 4295037296
  },
  [pscustomobject]@{
    Id = 'C6_k3'
    Shape = 'alternating'
    Budgets = @(27,81,243)
    Vars = 4096
    Samples = 4
    MinArity = 3
    MaxArity = 3
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'meet'
    VRoot = 'join'
    EnsureUniqueLeaves = $false
    SeedBase = 4295047296
  },
  [pscustomobject]@{
    Id = 'C6_k4'
    Shape = 'alternating'
    Budgets = @(32,128,256)
    Vars = 4096
    Samples = 4
    MinArity = 4
    MaxArity = 4
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'meet'
    VRoot = 'join'
    EnsureUniqueLeaves = $false
    SeedBase = 4295057296
  },
  [pscustomobject]@{
    Id = 'C6_k6'
    Shape = 'alternating'
    Budgets = @(36,216,432)
    Vars = 4096
    Samples = 4
    MinArity = 6
    MaxArity = 6
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'meet'
    VRoot = 'join'
    EnsureUniqueLeaves = $false
    SeedBase = 4295067296
  },
  [pscustomobject]@{
    Id = 'B1_bestcase'
    Shape = 'bestcase'
    Budgets = @(1..14)
    Vars = 4096
    Samples = 6
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295077296
    HighDepthStart = 12
    HighDepthSampleCap = 2
  },
  [pscustomobject]@{
    Id = 'B2_worstcase'
    Shape = 'worstcase'
    Budgets = @(1..14)
    Vars = 4096
    Samples = 2
    MinArity = 2
    MaxArity = 2
    PJoin = 0.5
    PAlt  = 1.0
    URoot = 'auto'
    VRoot = 'auto'
    EnsureUniqueLeaves = $false
    SeedBase = 4295087296
    HighDepthStart = 10
    HighDepthSampleCap = 1
  }
)

if ($Filter -and $Filter.Count -gt 0) {
  $experiments = $experiments | Where-Object { $Filter -contains $_.Id }
  if ($experiments.Count -eq 0) {
    Write-Warning 'Filter matched no experiments. Exiting.'
    return
  }
}

$reprModes = switch ($Representation) {
  'canonical' { @('canonical') }
  'legacy'    { @('legacy') }
  'both'      { @('canonical','legacy') }
}

Ensure-Dir $OutRoot
Ensure-Dir $BinDir

if ($Build -ne 'skip') {
  $built = $false
  try { $built = Try-Make } catch { $built = $false }
  if (-not $built -or -not (Test-Path $CheckExe) -or -not (Test-Path $GenExe)) {
    Gpp-Or-DIE
    if (-not $DryRun) {
      Write-Host '[build] compiling directly...'
  & g++ -O2 -std=c++17 -I include src/whitman.cpp src/freese.cpp src/cosmadakis.cpp src/hunt.cpp src/runner_min.cpp -o $CheckExe $linkArgs
  & g++ -O2 -std=c++17 -I include src/lattice_gen.cpp -o $GenExe $linkArgs
    } else {
      Write-Warning 'Dry-run requested and build artifacts missing; binaries will not be rebuilt.'
    }
  }
}

if (-not (Test-Path $CheckExe)) { throw "Missing $CheckExe" }
if (-not (Test-Path $GenExe))   { throw "Missing $GenExe" }

foreach ($repr in $reprModes) {
  $manifest = @()
  $expIndex = 0
  $reprOutRoot = Join-Path $OutRoot $repr
  Ensure-Dir $reprOutRoot

  foreach ($exp in $experiments) {
    $expIndex += 1
    $expDir = Join-Path $reprOutRoot $exp.Id
    Ensure-Dir $expDir

  $summary = @()
  $metaRows = @()
  $statsRows = @()
  $baseSamples = if ($SampleOverride.HasValue) { [int]$SampleOverride.Value } else { [int]$exp.Samples }
  $samplesPlan = @()

    $budgets = $exp.Budgets
    if ($BudgetCeiling -gt 0) {
      $budgets = $budgets | Where-Object { $_ -le $BudgetCeiling }
    }
    if ($AltfullBudgetCeiling -gt 0 -and $exp.Id -eq 'C1_altfull') {
      $budgets = $budgets | Where-Object { $_ -le $AltfullBudgetCeiling }
    }
    if (-not $budgets -or $budgets.Count -eq 0) {
      Write-Host ("[skip] {0} ({1} mode)" -f $exp.Id,$repr)
      continue
    }

    $seedOverrides = if ($exp.PSObject.Properties.Name -contains 'SeedOverrides') { $exp.SeedOverrides } else { $null }
    $seedBaseEffective = $SeedBase + ($expIndex * 10000)
    if ($exp.PSObject.Properties.Name -contains 'SeedBase' -and $exp.SeedBase) {
      $seedBaseEffective = [UInt64]$exp.SeedBase
    }
    $usedSeeds = @()

    foreach ($B in $budgets) {
      $samplesForBudget = $baseSamples
      if ($exp.PSObject.Properties.Name -contains 'HighDepthStart' -and $exp.PSObject.Properties.Name -contains 'HighDepthSampleCap') {
        if ($B -ge [int]$exp.HighDepthStart) {
          $samplesForBudget = [math]::Min($samplesForBudget, [int]$exp.HighDepthSampleCap)
        }
      }
      $samplesForBudget = [math]::Max(1, [int]$samplesForBudget)
      $samplesPlan += "${B}:$samplesForBudget"
      $seed = $null
      if ($seedOverrides) {
        $keyString = $B.ToString()
        if ($seedOverrides.ContainsKey($keyString)) {
          $seed = [UInt64]$seedOverrides[$keyString]
        } elseif ($seedOverrides.ContainsKey($B)) {
          $seed = [UInt64]$seedOverrides[$B]
        }
      }
      if ($null -eq $seed) {
        $seed = $seedBaseEffective + [UInt64]$B
      }
      $usedSeeds += "${B}:$seed"
      $vars = [int]$exp.Vars
      if ($exp.EnsureUniqueLeaves) {
        $vars = [math]::Max($vars, [math]::Pow(2, $B))
      }

      $jsonPath = Join-Path $expDir ("pairs_B{0}.jsonl" -f $B)
      $genArgs = @(
        'rand','--seed',$seed,'--vars',$vars,'--budget',$B,'--samples',$samplesForBudget,
        '--shape',$exp.Shape,'--min_arity',$exp.MinArity,'--max_arity',$exp.MaxArity,
        '--p_join',$exp.PJoin,'--p_alt',$exp.PAlt,'--out',$jsonPath
      )
      if ($exp.Shape -eq 'alternating' -and ($exp.URoot -ne 'auto' -or $exp.VRoot -ne 'auto')) {
        $genArgs += @('--u_root',$exp.URoot,'--v_root',$exp.VRoot)
      }

      Write-Host ("[gen] {0} B={1} seed={2} ({3})" -f $exp.Id,$B,$seed,$repr)
      if (-not $DryRun) { & $GenExe @genArgs }

      $peek = @()
      if (-not $DryRun) {
        $peek = (Get-Content $jsonPath -TotalCount 3) | ForEach-Object { $_ | ConvertFrom-Json }
        if ($peek) {
          $uHead = ($peek[0].u)
          $vHead = ($peek[0].v)
          $uOp = if ($uHead -and $uHead.Length -ge 2) { $uHead.Substring(1,1) } else { '?' }
          $vOp = if ($vHead -and $vHead.Length -ge 2) { $vHead.Substring(1,1) } else { '?' }
          Write-Host ("  [sanity] first pair ops: u='{0}' v='{1}'" -f $uOp,$vOp)
        }
      }

      $engineCsvs = @{}
      foreach ($e in $engines) {
        $csvPath = Join-Path $expDir ("{0}_B{1}.csv" -f $e,$B)
        Write-Host ("  [run] {0} B={1} ({2})" -f $e,$B,$repr)
        if (-not $DryRun) {
          & $CheckExe --engine $e --stats --repr $repr --json $jsonPath | Set-Content -Encoding ascii $csvPath
        }
        $engineCsvs[$e] = $csvPath
      }

      if ($DryRun) { continue }

      $lineObjs = Get-Content $jsonPath | ForEach-Object { $_ | ConvertFrom-Json }
      $pairDepths = @()
      $pairAlt = @()
      $pairShare = @()
      foreach ($line in $lineObjs) {
        if ($line.meta) {
          $pairDepths += [double][math]::Max($line.meta.u.height, $line.meta.v.height)
          $pairAlt += [double]([double]$line.meta.pair.avg_alt_index)
          $pairShare += [double]([double]$line.meta.pair.avg_share_ratio)
          $metaRows += [pscustomobject]@{
            experiment = $exp.Id
            budget = $B
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

      foreach ($e in $engines) {
        $rows = Import-Csv $engineCsvs[$e]
        $totals = $rows | ForEach-Object { [double]$_.total_us }
        $q = Quantiles([double[]]$totals)
        $avg = Mean([double[]]$totals)
        foreach ($row in $rows) {
          $uvStats = Parse-StatsJson($row.uv_stats)
          $vuStats = Parse-StatsJson($row.vu_stats)
          $uvObj = New-StatsObject($exp.Id,$exp.Shape,$B,$e,$row.pair_id,'uv',$uvStats)
          if ($uvObj) { $statsRows += $uvObj }
          $vuObj = New-StatsObject($exp.Id,$exp.Shape,$B,$e,$row.pair_id,'vu',$vuStats)
          if ($vuObj) { $statsRows += $vuObj }
        }
        $summary += [pscustomobject]@{
          experiment        = $exp.Id
          shape             = $exp.Shape
          budget            = $B
          engine            = $e
          median_pairDepth  = [double]$depthQ.p50
          p25_us            = [double]$q.p25
          median_us         = [double]$q.p50
          p75_us            = [double]$q.p75
          mean_us           = [double][math]::Round($avg,3)
          n_pairs           = $rows.Count
          median_alt_index  = [double][math]::Round($altQ.p50,6)
          median_share      = [double][math]::Round($shareQ.p50,6)
          csv_path          = [System.IO.Path]::GetFileName($engineCsvs[$e])
          json_path         = [System.IO.Path]::GetFileName($jsonPath)
        }
      }
    }

    if ($DryRun) { continue }

    $sumPath = Join-Path $expDir 'summary.csv'
    $summary | Sort-Object engine,budget | Export-Csv -NoTypeInformation -Encoding ascii $sumPath
    $metaPath = $null
    if ($metaRows.Count -gt 0) {
      $metaPath = Join-Path $expDir 'meta_pairs.csv'
      $metaRows | Export-Csv -NoTypeInformation -Encoding ascii $metaPath
    }

    $statsPath = $null
    $statsAggPath = $null
    if ($statsRows.Count -gt 0) {
      $statsPath = Join-Path $expDir 'solver_stats.csv'
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
        $statsAggPath = Join-Path $expDir 'solver_stats_agg.csv'
        $statAgg | Export-Csv -NoTypeInformation -Encoding ascii $statsAggPath
      }
    }

    $seedMode = if ($seedOverrides) { 'override' } else { 'auto' }
    $seedMapString = if ($seedOverrides) {
      ($seedOverrides.GetEnumerator() | Sort-Object Name | ForEach-Object {
        "{0}:{1}" -f $_.Name,$_.Value
      }) -join ';'
    } else { '' }
    $seedPlan = if ($usedSeeds) { $usedSeeds -join ';' } else { '' }

    $manifest += [pscustomobject]@{
      id = $exp.Id
      representation = $repr
      shape = $exp.Shape
      budgets = ($budgets -join ',')
      samples_requested = $exp.Samples
  samples_used = $baseSamples
  samples_plan = ($samplesPlan -join ';')
      vars = $exp.Vars
      base_seed = if ($seedOverrides) { '' } else { $seedBaseEffective }
      seed_mode = $seedMode
      seed_overrides = $seedMapString
      seed_plan = $seedPlan
      ensure_unique_leaves = $exp.EnsureUniqueLeaves
      output_dir = $expDir
      summary_csv = [System.IO.Path]::GetFileName($sumPath)
      meta_csv    = if ($metaPath) { [System.IO.Path]::GetFileName($metaPath) } else { '' }
      stats_csv   = if ($statsPath) { [System.IO.Path]::GetFileName($statsPath) } else { '' }
      stats_agg_csv = if ($statsAggPath) { [System.IO.Path]::GetFileName($statsAggPath) } else { '' }
    }
  }

  if ($DryRun) { continue }

  if ($manifest.Count -gt 0) {
    $manifestPath = Join-Path $reprOutRoot 'manifest.csv'
    $manifest | Export-Csv -NoTypeInformation -Encoding ascii $manifestPath
    Write-Host ("[ok] manifest ({0}) -> {1}" -f $repr,$manifestPath)
  }
}

if ($DryRun) { Write-Host '[dry-run] no files written.' }
