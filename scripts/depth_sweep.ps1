param(
  [UInt64]$Seed    = 42,
  [int]   $Vars    = 1000,                                 # use many vars to avoid accidental idempotent collapse
  [int[]] $Budgets = @(50,100,150,200,250,300),            # you can push higher once you see the curve
  [int]   $Samples = 150,
  [ValidateSet('alternating','leftspine')][string] $Shape = 'alternating',
  [ValidateSet('auto','skip')][string] $Build = 'auto',

  # new: control roots when Shape = 'alternating'
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

# fixed knobs to isolate *depth*
$minArity = 2; $maxArity = 2; $pJoin = 0.5; $pAlt = 1.0  # strict alternation, binary fanout

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

foreach ($B in $Budgets) {
  $data = Join-Path $OutDir ("depth_B{0}.jsonl" -f $B)
  Write-Host "[gen] B=$B -> $data"

  $genArgs = @(
    'rand','--seed', $Seed, '--vars', $Vars, '--budget', $B, '--samples', $Samples,
    '--shape', $Shape, '--min_arity', $minArity, '--max_arity', $maxArity,
    '--p_join', $pJoin, '--p_alt', $pAlt, '--out', $data
  )
  if ($Shape -eq 'alternating' -and ($URoot -ne 'auto' -or $VRoot -ne 'auto')) {
    $genArgs += @('--u_root', $URoot, '--v_root', $VRoot)
  }

  & $GenExe @genArgs

  $engs = @('whitman','freese','cosma','hunt')
  $csvs = @{}
  foreach ($e in $engs) {
    $csv = Join-Path $OutDir ("{0}_B{1}.csv" -f $e,$B)
    Write-Host "[run] $e B=$B"
    & $CheckExe --engine $e --json $data | Set-Content -Encoding ascii $csv
    $csvs[$e] = $csv
  }

  # If your generator doesn’t emit u_stats/v_stats, just use B as proxy for depth
  $jsonLines = Get-Content $data | ForEach-Object { $_ | ConvertFrom-Json }
  $pairDepths = @()
  foreach ($line in $jsonLines) {
    if ($line.PSObject.Properties.Name -contains 'u_stats' -and $line.PSObject.Properties.Name -contains 'v_stats') {
      $pairDepths += [double][math]::Max($line.u_stats.max_depth, $line.v_stats.max_depth)
    } else {
      $pairDepths += [double]$B
    }
  }
  $depthQ = Quantiles([double[]]$pairDepths)

  foreach ($e in $engs) {
    $rows = Import-Csv $csvs[$e]
    $totals = $rows | ForEach-Object { [double]$_.total_us }
    $q = Quantiles([double[]]$totals)
    $summary += [pscustomobject]@{
      budget           = $B
      median_pairDepth = [int]$depthQ.p50
      engine           = $e
      p25_us           = [int]$q.p25
      median_us        = [int]$q.p50
      p75_us           = [int]$q.p75
      n_pairs          = $rows.Count
    }
  }
}

$sumCsv = Join-Path $OutDir "depth_summary.csv"
$summary | Sort-Object engine,budget | Export-Csv -NoTypeInformation -Encoding ascii $sumCsv
Write-Host "`n[ok] summary -> $sumCsv"
Write-Host "Plot median_us vs budget. With --shape alternating --u_root meet --v_root join, Whitman should blow up; others ~polynomial."
