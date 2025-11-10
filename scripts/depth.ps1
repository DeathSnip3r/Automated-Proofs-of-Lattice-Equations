param(
  [UInt64]$Seed    = 42,
  [int]   $Vars    = 1000,                                 # many vars to avoid DAG collapse
  [int[]] $Budgets = @(1,2,3,4,5,6,7,8,9,10,11,12),        # for altfull: heights; for alternating/leftspine: size
  [int]   $Samples = 40,
  [ValidateSet('altfull','alternating','leftspine')][string] $Shape = 'altfull',
  [ValidateSet('auto','skip')][string] $Build = 'auto',

  # only used for Shape = 'alternating'
  [ValidateSet('auto','meet','join')] $URoot = 'meet',
  [ValidateSet('auto','meet','join')] $VRoot = 'join'
)

$ErrorActionPreference = 'Stop'
$OutDir   = "results\depth"
$BinDir   = "bin"
$CheckExe = Join-Path $BinDir "check.exe"
$GenExe   = Join-Path $BinDir "lattice_gen.exe"

function Ensure-Dir($p){ if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }
function Try-Make {
  foreach ($mk in @('make','mingw32-make')) {
    $cmd = Get-Command $mk -ErrorAction SilentlyContinue
    if ($cmd) { & $mk all; return $true }
  }
  return $false
}
function Gpp-Or-DIE { if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) { throw "g++ not found" } }

Ensure-Dir $OutDir
if ($Build -ne 'skip') {
  $ok = $false
  try { $ok = Try-Make } catch { $ok = $false }
  if (-not (Test-Path $CheckExe) -or -not (Test-Path $GenExe)) {
    Gpp-Or-DIE
    Write-Host "[build] compiling directly..."
    & g++ -O2 -std=c++17 -I include `
      src/whitman.cpp src/freese.cpp src/cosmadakis.cpp src/hunt.cpp src/runner_min.cpp `
      -o $CheckExe
    & g++ -O2 -std=c++17 -I include src/lattice_gen.cpp -o $GenExe
  }
}

if (-not (Test-Path $CheckExe)) { throw "Missing $CheckExe" }
if (-not (Test-Path $GenExe))   { throw "Missing $GenExe" }

# fixed knobs to isolate depth characteristics
$minArity = 2; $maxArity = 2; $pJoin = 0.5; $pAlt = 1.0

$summary = @()

function Quantiles([double[]]$xs) {
  if (-not $xs -or $xs.Count -eq 0) { return @{p25=0; p50=0; p75=0} }
  $ys = $xs | Sort-Object
  $n  = $ys.Count
  return @{
    p25 = $ys[[math]::Floor(0.25*($n-1))]
    p50 = $ys[[math]::Floor(0.50*($n-1))]
    p75 = $ys[[math]::Floor(0.75*($n-1))]
  }
}
function Mean([double[]]$xs) {
  if (-not $xs -or $xs.Count -eq 0) { return 0.0 }
  return ($xs | Measure-Object -Average).Average
}

$engs = @('whitman','freese','cosma','hunt')

foreach ($B in $Budgets) {
  $data = Join-Path $OutDir ("{0}_B{1}.jsonl" -f $Shape,$B)
  Write-Host "[gen] $Shape B=$B -> $data"

  $genArgs = @(
    'rand','--seed', $Seed, '--vars', $Vars, '--budget', $B, '--samples', $Samples,
    '--shape', $Shape, '--min_arity', $minArity, '--max_arity', $maxArity,
    '--p_join', $pJoin, '--p_alt', $pAlt, '--out', $data
  )
  if ($Shape -eq 'alternating' -and ($URoot -ne 'auto' -or $VRoot -ne 'auto')) {
    $genArgs += @('--u_root', $URoot, '--v_root', $VRoot)
  }

  & $GenExe @genArgs

  # --- sanity peek: DO NOT overwrite $URoot/$VRoot (case-insensitive!) ---
  $peek = (Get-Content $data -TotalCount 3) | ForEach-Object { $_ | ConvertFrom-Json }
  if ($peek) {
    $uHead = ($peek[0].u)
    $vHead = ($peek[0].v)

    if ($uHead -and $uHead.Length -ge 2) { $uOp = $uHead.Substring(1,1) } else { $uOp = '?' }
    if ($vHead -and $vHead.Length -ge 2) { $vOp = $vHead.Substring(1,1) } else { $vOp = '?' }

    Write-Host ("[sanity] first pair ops: u-op='{0}' v-op='{1}' (expect '*' for u and '+' for v in hard regime)" -f $uOp,$vOp)
  }

  # --- run all engines ---
  $csvs = @{}
  foreach ($e in $engs) {
    $csv = Join-Path $OutDir ("{0}_{1}_B{2}.csv" -f $Shape,$e,$B)
    Write-Host "[run] $e B=$B"
  & $CheckExe --engine $e --stats --json $data | Set-Content -Encoding ascii $csv
    $csvs[$e] = $csv
  }

  # --- agreement check across engines for this budget ---
  $maps = @{}
  foreach ($e in $engs) {
    $rows = Import-Csv $csvs[$e]
    $m = @{}
    foreach ($r in $rows) { $m[[int]$r.pair_id] = $r }
    $maps[$e] = $m
  }

  $pairIds = ($maps['whitman'].Keys | Sort-Object)
  $bad = @()
  foreach ($pairId in $pairIds) {
    $sigs = @{}
    foreach ($e in $engs) {
      $r = $maps[$e][$pairId]
      $sigs["$($r.u_leq_v),$($r.v_leq_u)"] = $true
    }
    if ($sigs.Keys.Count -ne 1) {
      $bad += [pscustomobject]@{
        pair_id = $pairId
        whitman = "$($maps['whitman'][$pairId].u_leq_v),$($maps['whitman'][$pairId].v_leq_u)"
        freese  = "$($maps['freese'][$pairId].u_leq_v),$($maps['freese'][$pairId].v_leq_u)"
        cosma   = "$($maps['cosma'][$pairId].u_leq_v),$($maps['cosma'][$pairId].v_leq_u)"
        hunt    = "$($maps['hunt'][$pairId].u_leq_v),$($maps['hunt'][$pairId].v_leq_u)"
      }
    }
  }

  if ($bad.Count -gt 0) {
    $disCsv = Join-Path $OutDir ("{0}_disagreements_B{1}.csv" -f $Shape,$B)
    $bad | Export-Csv -NoTypeInformation -Encoding ascii $disCsv
    Write-Host ("[warn] Disagreements at B={0}: {1} (saved {2})" -f $B, $bad.Count, $disCsv)
  } else {
    Write-Host ("[ok] All engines agree at B={0}" -f $B)
  }

  # Depth proxy: use stats if present; else use B
  $jsonLines = Get-Content $data | ForEach-Object { $_ | ConvertFrom-Json }
  $pairDepths = @()
  $pairAltIdx = @()
  $pairShareRatio = @()
  foreach ($line in $jsonLines) {
    if ($line.meta) {
      if ($line.meta.u -and $line.meta.v) {
        $pairDepths += [double][math]::Max($line.meta.u.height, $line.meta.v.height)
        $pairAltIdx += [double]([math]::Round([double]$line.meta.pair.avg_alt_index,6))
        $pairShareRatio += [double]([math]::Round([double]$line.meta.pair.avg_share_ratio,6))
        continue
      }
    }
    $pairDepths += [double]$B
  }
  $depthQ = Quantiles([double[]]$pairDepths)
  $altQ = if ($pairAltIdx.Count -gt 0) { Quantiles([double[]]$pairAltIdx) } else { @{p25=0;p50=0;p75=0} }
  $shareQ = if ($pairShareRatio.Count -gt 0) { Quantiles([double[]]$pairShareRatio) } else { @{p25=0;p50=0;p75=0} }

  foreach ($e in $engs) {
    $rows = Import-Csv $csvs[$e]
    $totals = $rows | ForEach-Object { [double]$_.total_us }
    $q = Quantiles([double[]]$totals)
    $avg = Mean([double[]]$totals)
    $summary += [pscustomobject]@{
      shape            = $Shape
      budget           = $B
      median_pairDepth = [int]$depthQ.p50
      engine           = $e
      p25_us           = [int]$q.p25
      median_us        = [int]$q.p50
      p75_us           = [int]$q.p75
      mean_us          = [int][math]::Round($avg)
      n_pairs          = $rows.Count
      median_alt_index = [double][math]::Round($altQ.p50,6)
      median_share     = [double][math]::Round($shareQ.p50,6)
    }
  }
}

$sumCsv = Join-Path $OutDir ("{0}_summary.csv" -f $Shape)
$summary | Sort-Object engine,budget | Export-Csv -NoTypeInformation -Encoding ascii $sumCsv
Write-Host ("`n[ok] summary -> {0}" -f $sumCsv)

# ---------------- Agreement sanity across all budgets ----------------
$disagreements = 0
$sampleDisagreements = @()

foreach ($B in $Budgets) {
  $rowsByEngine = @{}
  foreach ($e in $engs) {
    $csvPath = Join-Path $OutDir ("{0}_{1}_B{2}.csv" -f $Shape,$e,$B)
    $rowsByEngine[$e] = Import-Csv $csvPath
  }

  $pairIdsAll = $rowsByEngine['whitman'] | ForEach-Object { [int]$_.pair_id }
  foreach ($pairId in $pairIdsAll) {
    $sigSet = @()
    foreach ($e in $engs) {
      $row = $rowsByEngine[$e] | Where-Object { [int]$_.pair_id -eq $pairId } | Select-Object -First 1
      if ($null -ne $row) {
        $sigSet += ("{0},{1}" -f $row.u_leq_v, $row.v_leq_u)
      }
    }
    $unique = $sigSet | Select-Object -Unique
    if ($unique.Count -gt 1) {
      $disagreements++
      if ($sampleDisagreements.Count -lt 5) {
        $sampleDisagreements += [pscustomobject]@{
          Budget = $B
          Pair   = $pairId
          Sigs   = ($sigSet -join ' | ')
        }
      }
    }
  }
}

if ($disagreements -gt 0) {
  Write-Host ("[agreement] disagreements found: {0}" -f $disagreements)
  $sampleDisagreements | Format-Table -AutoSize | Out-String | Write-Host
} else {
  Write-Host "[agreement] all engines agree across all budgets."
}

# ---------------- Small growth-factor readout (median per step) --------
$growth = @()
foreach ($e in $engs) {
  $rowsE = $summary | Where-Object { $_.engine -eq $e } | Sort-Object budget
  $first = $rowsE | Where-Object { $_.median_us -gt 0 } | Select-Object -First 1
  $last  = $rowsE | Select-Object -Last 1
  if ($first -and $last -and $first.median_us -gt 0) {
    $steps = [double]($last.budget - $first.budget)
    if ($steps -gt 0) {
      $g = [Math]::Pow(($last.median_us / [double]$first.median_us), 1.0 / $steps)
      $growth += [pscustomobject]@{
        engine   = $e
        from     = $first.budget
        to       = $last.budget
        per_step = [Math]::Round($g, 3)
      }
    }
  }
}
if ($growth.Count -gt 0) {
  Write-Host "[growth] approx median multiplicative growth per height/size step:"
  $growth | Format-Table -AutoSize | Out-String | Write-Host
}

Write-Host ("`nPlot median_us (and/or mean_us) vs {0}. Whitman should blow up for altfull (height); others ~polynomial." -f ($(if($Shape -eq 'altfull'){'height'}else{'budget'})))
