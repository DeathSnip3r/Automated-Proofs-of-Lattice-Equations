#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ENGINES = ["whitman", "freese", "cosma", "hunt"]
NON_C1_SAMPLE_DEFAULT = 2
CONTROLLED_DEFAULT_OUT = os.path.join("results", "controlled")
STRESS_DEFAULT_OUT = os.path.join("results", "stress")
BIN_DIR = "bin"
IS_WINDOWS = os.name == "nt"
CHECK_NAME = "check.exe" if IS_WINDOWS else "check"
GEN_NAME = "lattice_gen.exe" if IS_WINDOWS else "lattice_gen"
CHECK_EXE = os.path.join(BIN_DIR, CHECK_NAME)
GEN_EXE = os.path.join(BIN_DIR, GEN_NAME)


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> None:
    log(f"exec: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def try_make(project_root: Path) -> bool:
    for mk in ("make",):
        if shutil_which(mk):
            try:
                run_command([mk, "all"], cwd=project_root)
                return True
            except subprocess.CalledProcessError:
                log(f"warning: build tool '{mk}' failed")
    return False


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def build_binaries(project_root: Path, build_mode: str, dry_run: bool) -> None:
    if build_mode == "skip":
        return
    built = False
    try:
        built = try_make(project_root)
    except Exception as exc:  # pragma: no cover
        log(f"warning: make failed ({exc})")
        built = False
    check_path = project_root / CHECK_EXE
    gen_path = project_root / GEN_EXE
    if built and check_path.exists() and gen_path.exists():
        return
    if dry_run:
        log("warning: dry-run requested and binaries missing; build skipped")
        return
    log("[build] compiling directly via g++")
    include_flag = ["-I", "include"]
    # Compile without optimizations for empirical evaluation
    compile_flags = ["-std=c++17", "-g"] 
    run_command(
        [
            "g++",
            *compile_flags,
            *include_flag,
            "src/whitman.cpp",
            "src/freese.cpp",
            "src/cosmadakis.cpp",
            "src/hunt.cpp",
            "src/runner_min.cpp",
            "-o",
            str(check_path),
        ],
        cwd=project_root,
    )
    run_command(
        [
            "g++",
            *compile_flags,
            *include_flag,
            "src/lattice_gen.cpp",
            "-o",
            str(gen_path),
        ],
        cwd=project_root,
    )


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    nums = sorted(values)
    if not nums:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    n = len(nums)
    return {
        "p25": nums[math.floor(0.25 * (n - 1))],
        "p50": nums[math.floor(0.50 * (n - 1))],
        "p75": nums[math.floor(0.75 * (n - 1))],
    }


def mean(values: Iterable[float]) -> float:
    nums = list(values)
    if not nums:
        return 0.0
    return sum(nums) / float(len(nums))


def parse_stats_json(cell: str) -> Optional[Dict[str, Any]]:
    if not cell:
        return None
    clean = cell.strip()
    if not clean:
        return None
    if clean.startswith('"') and clean.endswith('"'):
        clean = clean[1:-1]
    clean = clean.replace('""', '"')
    clean = clean.strip()
    if not clean:
        return None
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        log(f"warning: failed to parse stats json: {clean}")
        return None


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ControlledExperiment:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def seed_overrides(self) -> Optional[Dict[str, int]]:
        return self.data.get("seed_overrides")

    def has_seed_base(self) -> bool:
        return "seed_base" in self.data and self.data["seed_base"] is not None

    def seed_base(self) -> Optional[int]:
        return self.data.get("seed_base")


class StressExperiment:
    def __init__(self, data: Dict[str, Any]):
        self.data = data


CONTROLLED_EXPERIMENTS: List[ControlledExperiment] = [
    ControlledExperiment(
        {
            "id": "C1_altfull",
            "shape": "altfull",
            "budgets": list(range(1, 17)),
            "vars": 4096,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": True,
            "seed_base": 4294977296,
            "sample_policy": "base_with_cap",
            "high_depth_threshold": 12,
            "high_depth_cap": 2,
            "allow_global_override": False,
        }
    ),
    ControlledExperiment(
        {
            "id": "E_bestcase",
            "shape": "bestcase",
            "budgets": list(range(1, 17)),
            "vars": 4096,
            "samples": 16,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": True,
            "seed_base": 6000000000,
            "sample_policy": "base",
            "allow_global_override": True,
        }
    ),
    ControlledExperiment(
        {
            "id": "E_worstcase",
            "shape": "worstcase",
            "budgets": list(range(1, 17)),
            "vars": 128,
            "samples": 16,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 6000100000,
            "sample_policy": "base_with_cap",
            "high_depth_threshold": 12,
            "high_depth_cap": 2,
            "allow_global_override": False,
        }
    ),
    ControlledExperiment(
        {
            "id": "C2_altfixed",
            "shape": "alternating",
            "budgets": [50, 100, 200, 400, 800],
            "vars": 2048,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "meet",
            "v_root": "join",
            "ensure_unique_leaves": False,
            "seed_base": 4294987296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C3_altpolarity",
            "shape": "alternating",
            "budgets": [50, 100, 200, 400, 800],
            "vars": 2048,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "join",
            "v_root": "meet",
            "ensure_unique_leaves": False,
            "seed_base": 4294997296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C4_share_low",
            "shape": "balanced",
            "budgets": [200, 400, 600, 800, 1000],
            "vars": 64,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 0.2,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 4295007296,
            "seed_overrides": {"200": 8000, "400": 8300, "600": 8350, "800": 8400, "1000": 8450},
        }
    ),
    ControlledExperiment(
        {
            "id": "C4_share_medium",
            "shape": "balanced",
            "budgets": [200, 400, 600, 800, 1000],
            "vars": 256,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 0.6,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 4295008296,
            "seed_overrides": {"200": 8900, "400": 9001, "600": 9051, "800": 9101, "1000": 9151},
        }
    ),
    ControlledExperiment(
        {
            "id": "C4_share_high",
            "shape": "balanced",
            "budgets": [200, 400, 600, 800, 1000],
            "vars": 4096,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 0.8,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 4295009296,
            "seed_overrides": {"200": 7019, "400": 8700, "600": 8750, "800": 8800, "1000": 8850},
        }
    ),
    ControlledExperiment(
        {
            "id": "C5_spines",
            "shape": "leftspine",
            "budgets": [20, 40, 60, 80],
            "vars": 1024,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 4295017296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C5_spines_r",
            "shape": "rightspine",
            "budgets": [20, 40, 60, 80],
            "vars": 1024,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "auto",
            "v_root": "auto",
            "ensure_unique_leaves": False,
            "seed_base": 4295027296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C6_k2",
            "shape": "alternating",
            "budgets": [40, 80, 160, 320],
            "vars": 4096,
            "samples": 32,
            "min_arity": 2,
            "max_arity": 2,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "meet",
            "v_root": "join",
            "ensure_unique_leaves": False,
            "seed_base": 4295037296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C6_k3",
            "shape": "alternating",
            "budgets": [27, 81, 243, 729],
            "vars": 4096,
            "samples": 32,
            "min_arity": 3,
            "max_arity": 3,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "meet",
            "v_root": "join",
            "ensure_unique_leaves": False,
            "seed_base": 4295047296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C6_k4",
            "shape": "alternating",
            "budgets": [32, 128, 256, 512],
            "vars": 4096,
            "samples": 32,
            "min_arity": 4,
            "max_arity": 4,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "meet",
            "v_root": "join",
            "ensure_unique_leaves": False,
            "seed_base": 4295057296,
        }
    ),
    ControlledExperiment(
        {
            "id": "C6_k6",
            "shape": "alternating",
            "budgets": [36, 216, 432, 648],
            "vars": 4096,
            "samples": 32,
            "min_arity": 6,
            "max_arity": 6,
            "p_join": 0.5,
            "p_alt": 1.0,
            "u_root": "meet",
            "v_root": "join",
            "ensure_unique_leaves": False,
            "seed_base": 4295067296,
        }
    ),
]


STRESS_EXPERIMENTS: List[StressExperiment] = []

STRESS_EXPERIMENTS.append(
    StressExperiment(
        {
            "id": "H1_balanced_noise",
            "mode": "standard",
            "shape": "balanced",
            "shape_label": "balanced",
            "budgets": [200],
            "vars": 8192,
            "samples": 256,
            "min_arity": 2,
            "max_arity": 6,
            "p_join": 0.5,
            "p_alt": 0.8,
            "ensure_unique_leaves": False,
            "u_root": "auto",
            "v_root": "auto",
        }
    )
)

for pj in (0.3, 0.5, 0.7):
    for pa in (0.2, 0.5, 0.8):
        STRESS_EXPERIMENTS.append(
            StressExperiment(
                {
                    "id": f"H2_rand_pj{int(round(pj * 100))}_pa{int(round(pa * 100))}",
                    "mode": "standard",
                    "shape": "balanced",
                    "shape_label": "rand",
                    "budgets": [50],
                    "vars": 4096,
                    "samples": 256,
                    "min_arity": 2,
                    "max_arity": 6,
                    "p_join": float(pj),
                    "p_alt": float(pa),
                    "ensure_unique_leaves": False,
                    "u_root": "auto",
                    "v_root": "auto",
                }
            )
        )

STRESS_EXPERIMENTS.append(
    StressExperiment(
        {
            "id": "H3_hybrid_bag",
            "mode": "bag",
            "shape": "hybrid",
            "shape_label": "bag",
            "budgets": [100],
            "vars": 4096,
            "samples": 256,
            "min_arity": 2,
            "max_arity": 6,
            "p_join": 0.5,
            "p_alt": 0.6,
            "ensure_unique_leaves": False,
            "u_root": "auto",
            "v_root": "auto",
            "bag_shapes": ["balanced", "alternating", "leftspine"],
            "bag_weights": [],
        }
    )
)


def filter_experiments(exp_list: List[Any], filters: List[str]) -> List[Any]:
    if not filters:
        return exp_list
    filtered = [exp for exp in exp_list if exp.data["id"] in filters]
    return filtered


def run_generator(project_root: Path, exe: Path, args: List[str]) -> None:
    run_command([str(exe)] + args, cwd=project_root)


def run_solver(project_root: Path, exe: Path, args: List[str], output_path: Path) -> None:
    with output_path.open("w", encoding="ascii", newline="") as handle:
        proc = subprocess.Popen(
            [str(exe)] + args,
            cwd=str(project_root),
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            if stderr:
                log(stderr.strip())
            raise subprocess.CalledProcessError(proc.returncode, proc.args)


def read_json_lines(path: Path) -> List[Dict[str, Any]]:
    lines = []
    if not path.exists():
        return lines
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def compute_samples_for_budget(
    exp: ControlledExperiment,
    budget: int,
    base_samples: int,
    override_samples: int,
    args: argparse.Namespace,
) -> int:
    policy = exp.data.get("sample_policy", "override")
    allow_override = exp.data.get("allow_global_override", True)
    has_override = args.sample_override is not None
    if policy == "override":
        return override_samples
    if policy == "base":
        if has_override and allow_override:
            return override_samples
        return base_samples
    if policy == "base_with_cap":
        threshold = exp.data.get("high_depth_threshold")
        if threshold is None:
            threshold = args.high_depth_threshold
        cap = exp.data.get("high_depth_cap")
        if cap is None:
            cap = args.high_depth_cap
        if budget >= threshold:
            limit = cap
            if has_override:
                limit = min(limit, override_samples)
            return min(base_samples, limit)
        if has_override and allow_override:
            return override_samples
        return base_samples
    # Fallback to override behaviour
    return override_samples


def controlled_suite(
    project_root: Path,
    args: argparse.Namespace,
    representations: List[str],
    filters: List[str],
) -> None:
    experiments = filter_experiments(CONTROLLED_EXPERIMENTS, filters)
    if not experiments:
        log("controlled: no experiments selected; skipping")
        return
    out_root = (project_root / Path(args.out_root or CONTROLLED_DEFAULT_OUT)).resolve()
    ensure_dir(out_root)
    ensure_dir(project_root / BIN_DIR)
    check_path = project_root / CHECK_EXE
    gen_path = project_root / GEN_EXE
    if not check_path.exists() or not gen_path.exists():
        raise RuntimeError(
            f"controlled: missing {CHECK_NAME} or {GEN_NAME}"
        )
    for repr_mode in representations:
        repr_out = out_root / repr_mode
        ensure_dir(repr_out)
        manifest_rows: List[Dict[str, Any]] = []
        for idx, exp in enumerate(experiments, start=1):
            exp_id = exp.data["id"]
            exp_dir = repr_out / exp_id
            ensure_dir(exp_dir)
            log(f"[controlled] experiment={exp_id} repr={repr_mode}")
            summary_rows: List[Dict[str, Any]] = []
            meta_rows: List[Dict[str, Any]] = []
            stats_rows: List[Dict[str, Any]] = []
            budgets = list(exp.data["budgets"])
            if args.budget_ceiling is not None:
                budgets = [b for b in budgets if b <= args.budget_ceiling]
            if args.altfull_ceiling is not None and exp_id == "C1_altfull":
                budgets = [b for b in budgets if b <= args.altfull_ceiling]
            if not budgets:
                log(f"  [skip] budgets empty after ceiling for {exp_id}")
                continue
            base_samples = int(exp.data["samples"])
            override_samples = (
                int(args.sample_override)
                if args.sample_override is not None
                else int(exp.data.get("default_override_samples", NON_C1_SAMPLE_DEFAULT))
            )
            policy = exp.data.get("sample_policy", "override")
            samples_default = base_samples if policy != "override" else override_samples
            used_seeds: List[str] = []
            sample_plan: List[str] = []
            seed_overrides = exp.seed_overrides()
            base_seed = exp.seed_base() if exp.has_seed_base() else args.seed_base
            for budget in budgets:
                samples_for_budget = compute_samples_for_budget(
                    exp,
                    budget,
                    base_samples,
                    override_samples,
                    args,
                )
                seed: Optional[int] = None
                if seed_overrides:
                    if str(budget) in seed_overrides:
                        seed = int(seed_overrides[str(budget)])
                    elif budget in seed_overrides:
                        seed = int(seed_overrides[budget])
                if seed is None:
                    seed = int(base_seed) + idx * 10000 + int(budget)
                used_seeds.append(f"{budget}:{seed}")
                vars_count = int(exp.data["vars"])
                if exp.data.get("ensure_unique_leaves"):
                    vars_count = max(vars_count, int(2 ** budget))
                json_path = exp_dir / f"pairs_B{budget}.jsonl"
                rel_json = os.path.relpath(json_path, project_root)
                sample_plan.append(f"{budget}:{samples_for_budget}")
                gen_args = [
                    "rand",
                    "--seed",
                    str(seed),
                    "--vars",
                    str(vars_count),
                    "--budget",
                    str(budget),
                    "--samples",
                    str(samples_for_budget),
                    "--shape",
                    exp.data["shape"],
                    "--min_arity",
                    str(exp.data["min_arity"]),
                    "--max_arity",
                    str(exp.data["max_arity"]),
                    "--p_join",
                    str(exp.data["p_join"]),
                    "--p_alt",
                    str(exp.data["p_alt"]),
                    "--out",
                    rel_json,
                ]
                if exp.data["shape"] == "alternating" and (
                    exp.data.get("u_root") != "auto" or exp.data.get("v_root") != "auto"
                ):
                    gen_args.extend(["--u_root", str(exp.data.get("u_root", "auto"))])
                    gen_args.extend(["--v_root", str(exp.data.get("v_root", "auto"))])
                log(
                    f"  [gen] {exp_id} B={budget} seed={seed} repr={repr_mode} samples={samples_for_budget}"
                )
                if not args.dry_run:
                    run_command([str(gen_path)] + gen_args, cwd=project_root)
                    peek_lines = read_json_lines(json_path)[:3]
                    if peek_lines:
                        u_head = peek_lines[0].get("u", "")
                        v_head = peek_lines[0].get("v", "")
                        u_op = u_head[1] if isinstance(u_head, str) and len(u_head) >= 2 else "?"
                        v_op = v_head[1] if isinstance(v_head, str) and len(v_head) >= 2 else "?"
                        log(f"    [sanity] first pair ops: u='{u_op}' v='{v_op}'")
                engine_csv: Dict[str, Path] = {}
                for engine in ENGINES:
                    csv_path = exp_dir / f"{engine}_B{budget}.csv"
                    log(f"    [run] engine={engine} budget={budget} repr={repr_mode}")
                    if not args.dry_run:
                        run_solver(
                            project_root,
                            check_path,
                            [
                                "--engine",
                                engine,
                                "--stats",
                                "--repr",
                                repr_mode,
                                "--json",
                                rel_json,
                            ],
                            csv_path,
                        )
                    engine_csv[engine] = csv_path
                if args.dry_run:
                    continue
                line_objs = read_json_lines(json_path)
                pair_depths: List[float] = []
                pair_alt: List[float] = []
                pair_share: List[float] = []
                for line in line_objs:
                    meta = line.get("meta")
                    if meta:
                        u_meta = meta.get("u", {})
                        v_meta = meta.get("v", {})
                        pair_depths.append(float(max(u_meta.get("height", budget), v_meta.get("height", budget))))
                        pair_alt.append(float(meta.get("pair", {}).get("avg_alt_index", 0.0)))
                        pair_share.append(float(meta.get("pair", {}).get("avg_share_ratio", 0.0)))
                        meta_rows.append(
                            {
                                "experiment": exp_id,
                                "budget": budget,
                                "u_nodes": int(u_meta.get("nodes", 0)),
                                "v_nodes": int(v_meta.get("nodes", 0)),
                                "u_height": int(u_meta.get("height", 0)),
                                "v_height": int(v_meta.get("height", 0)),
                                "avg_alt_index": float(meta.get("pair", {}).get("avg_alt_index", 0.0)),
                                "avg_share_ratio": float(meta.get("pair", {}).get("avg_share_ratio", 0.0)),
                            }
                        )
                    else:
                        pair_depths.append(float(budget))
                depth_q = quantiles(pair_depths)
                alt_q = quantiles(pair_alt)
                share_q = quantiles(pair_share)
                for engine in ENGINES:
                    rows = []
                    with engine_csv[engine].open("r", encoding="ascii") as handle:
                        writer = csv.DictReader(handle)
                        rows = list(writer)
                    totals = [float(row.get("total_us", 0.0)) for row in rows]
                    q = quantiles(totals)
                    avg = mean(totals)
                    for row in rows:
                        uv_stats = parse_stats_json(row.get("uv_stats", ""))
                        vu_stats = parse_stats_json(row.get("vu_stats", ""))
                        if uv_stats:
                            uv_entry = {
                                "experiment": exp_id,
                                "shape": exp.data["shape"],
                                "budget": budget,
                                "engine": engine,
                                "pair_id": int(row.get("pair_id", 0)),
                                "direction": "uv",
                            }
                            uv_entry.update(uv_stats)
                            stats_rows.append(uv_entry)
                        if vu_stats:
                            vu_entry = {
                                "experiment": exp_id,
                                "shape": exp.data["shape"],
                                "budget": budget,
                                "engine": engine,
                                "pair_id": int(row.get("pair_id", 0)),
                                "direction": "vu",
                            }
                            vu_entry.update(vu_stats)
                            stats_rows.append(vu_entry)
                    summary_rows.append(
                        {
                            "experiment": exp_id,
                            "shape": exp.data["shape"],
                            "budget": budget,
                            "engine": engine,
                            "median_pairDepth": depth_q["p50"],
                            "p25_us": q["p25"],
                            "median_us": q["p50"],
                            "p75_us": q["p75"],
                            "mean_us": round(avg, 3),
                            "n_pairs": len(rows),
                            "median_alt_index": round(alt_q["p50"], 6),
                            "median_share": round(share_q["p50"], 6),
                            "csv_path": engine_csv[engine].name,
                            "json_path": json_path.name,
                        }
                    )
                
            if args.dry_run:
                continue
            sum_path = exp_dir / "summary.csv"
            meta_path = exp_dir / "meta_pairs.csv"
            stats_path = exp_dir / "solver_stats.csv"
            stats_agg_path = exp_dir / "solver_stats_agg.csv"
            summary_rows.sort(key=lambda r: (r["engine"], r["budget"]))
            write_csv(sum_path, summary_rows)
            if meta_rows:
                write_csv(meta_path, meta_rows)
            if stats_rows:
                write_csv(stats_path, stats_rows)
                base_fields = {"experiment", "shape", "budget", "engine", "pair_id", "direction"}
                metric_names = set()
                for row in stats_rows:
                    metric_names.update(set(row.keys()) - base_fields)
                grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
                for row in stats_rows:
                    key = (
                        row["experiment"],
                        row["shape"],
                        row["budget"],
                        row["engine"],
                        row["direction"],
                    )
                    grouped[key].append(row)
                agg_rows: List[Dict[str, Any]] = []
                for key, rows in grouped.items():
                    entry = {
                        "experiment": key[0],
                        "shape": key[1],
                        "budget": key[2],
                        "engine": key[3],
                        "direction": key[4],
                        "n": len(rows),
                    }
                    for metric in sorted(metric_names):
                        vals = []
                        for row in rows:
                            if metric in row:
                                try:
                                    vals.append(float(row[metric]))
                                except ValueError:
                                    continue
                        entry[f"avg_{metric}"] = mean(vals) if vals else 0.0
                    agg_rows.append(entry)
                agg_rows.sort(key=lambda r: (r["engine"], r["budget"], r["direction"]))
                if agg_rows:
                    write_csv(stats_agg_path, agg_rows)
            manifest_rows.append(
                {
                    "id": exp_id,
                    "representation": repr_mode,
                    "shape": exp.data["shape"],
                    "budgets": ",".join(str(b) for b in budgets),
                    "samples_requested": exp.data["samples"],
                    "samples_used": samples_default,
                    "vars": exp.data["vars"],
                    "base_seed": "" if exp.seed_overrides() else base_seed,
                    "seed_mode": "override" if exp.seed_overrides() else "auto",
                    "seed_overrides": ";".join(
                        f"{k}:{v}" for k, v in (exp.seed_overrides() or {}).items()
                    )
                    if exp.seed_overrides()
                    else "",
                    "seed_plan": ";".join(used_seeds),
                    "sample_plan": ";".join(sample_plan),
                    "ensure_unique_leaves": exp.data.get("ensure_unique_leaves", False),
                    "output_dir": os.path.relpath(exp_dir, project_root),
                    "summary_csv": sum_path.name,
                    "meta_csv": meta_path.name if meta_rows else "",
                    "stats_csv": stats_path.name if stats_rows else "",
                    "stats_agg_csv": stats_agg_path.name if stats_rows else "",
                }
            )
        if args.dry_run:
            continue
        if manifest_rows:
            manifest_path = repr_out / "manifest.csv"
            write_csv(manifest_path, manifest_rows)
            log(f"[controlled] manifest -> {manifest_path}")
    if args.dry_run:
        log("[controlled] dry-run complete; no files written")


def allocate_bag_samples(total: int, weights: List[int], shapes: List[str]) -> List[int]:
    if not shapes:
        return []
    if not weights or len(weights) != len(shapes) or sum(weights) <= 0:
        weights = [1] * len(shapes)
    alloc = [0] * len(shapes)
    weight_sum = sum(weights)
    assigned = 0
    for i, weight in enumerate(weights):
        portion = int(math.floor(total * weight / weight_sum))
        alloc[i] = portion
        assigned += portion
    remainder = total - assigned
    idx = 0
    while remainder > 0 and shapes:
        alloc[idx % len(shapes)] += 1
        remainder -= 1
        idx += 1
    return alloc


def stress_suite(
    project_root: Path,
    args: argparse.Namespace,
    representations: List[str],
    filters: List[str],
) -> None:
    experiments = filter_experiments(STRESS_EXPERIMENTS, filters)
    if not experiments:
        log("stress: no experiments selected; skipping")
        return
    out_root = (project_root / Path(args.out_root or STRESS_DEFAULT_OUT)).resolve()
    ensure_dir(out_root)
    ensure_dir(project_root / BIN_DIR)
    check_path = project_root / CHECK_EXE
    gen_path = project_root / GEN_EXE
    if not check_path.exists() or not gen_path.exists():
        raise RuntimeError(
            f"stress: missing {CHECK_NAME} or {GEN_NAME}"
        )
    for repr_mode in representations:
        repr_out = out_root / repr_mode
        ensure_dir(repr_out)
        manifest_rows: List[Dict[str, Any]] = []
        for idx, exp in enumerate(experiments, start=1):
            exp_id = exp.data["id"]
            exp_dir = repr_out / exp_id
            ensure_dir(exp_dir)
            log(f"[stress] experiment={exp_id} repr={repr_mode}")
            summary_rows: List[Dict[str, Any]] = []
            meta_rows: List[Dict[str, Any]] = []
            stats_rows: List[Dict[str, Any]] = []
            samples_default = args.stress_sample_override
            budgets = list(exp.data["budgets"])
            if args.budget_ceiling is not None:
                budgets = [b for b in budgets if b <= args.budget_ceiling]
            if not budgets:
                log(f"  [skip] budgets empty after ceiling for {exp_id}")
                continue
            for budget in budgets:
                seed = int(args.stress_seed_base) + idx * 10000 + int(budget)
                vars_count = int(exp.data["vars"])
                json_path = exp_dir / f"pairs_B{budget}.jsonl"
                rel_json = os.path.relpath(json_path, project_root)
                if exp.data["mode"] == "bag":
                    shapes = exp.data.get("bag_shapes", [])
                    alloc = allocate_bag_samples(samples_default, exp.data.get("bag_weights", []), shapes)
                    log(
                        f"  [gen-bag] {exp_id} budget={budget} repr={repr_mode} seed={seed} allocations={alloc}"
                    )
                    if not args.dry_run:
                        bag_files: List[Path] = []
                        for i, shape in enumerate(shapes):
                            count = alloc[i]
                            if count <= 0:
                                continue
                            shape_seed = seed + (i + 1) * 1000
                            tmp_path = Path(f"{json_path}.tmp_{i}")
                            gen_args = [
                                "rand",
                                "--seed",
                                str(shape_seed),
                                "--vars",
                                str(vars_count),
                                "--budget",
                                str(budget),
                                "--samples",
                                str(count),
                                "--shape",
                                shape,
                                "--min_arity",
                                str(exp.data["min_arity"]),
                                "--max_arity",
                                str(exp.data["max_arity"]),
                                "--p_join",
                                str(exp.data["p_join"]),
                                "--p_alt",
                                str(exp.data["p_alt"]),
                                "--out",
                                os.path.relpath(tmp_path, project_root),
                            ]
                            run_command(
                                [str(gen_path)] + gen_args,
                                cwd=project_root,
                            )
                            bag_files.append(tmp_path)
                        with json_path.open("w", encoding="ascii") as handle_out:
                            for tmp in bag_files:
                                with tmp.open("r", encoding="utf-8") as handle_in:
                                    for line in handle_in:
                                        handle_out.write(line)
                                tmp.unlink()
                else:
                    log(
                        f"  [gen] {exp_id} budget={budget} repr={repr_mode} seed={seed} samples={samples_default}"
                    )
                    if not args.dry_run:
                        gen_args = [
                            "rand",
                            "--seed",
                            str(seed),
                            "--vars",
                            str(vars_count),
                            "--budget",
                            str(budget),
                            "--samples",
                            str(samples_default),
                            "--shape",
                            exp.data["shape"],
                            "--min_arity",
                            str(exp.data["min_arity"]),
                            "--max_arity",
                            str(exp.data["max_arity"]),
                            "--p_join",
                            str(exp.data["p_join"]),
                            "--p_alt",
                            str(exp.data["p_alt"]),
                            "--out",
                            rel_json,
                        ]
                        run_command([str(gen_path)] + gen_args, cwd=project_root)
                engine_csv: Dict[str, Path] = {}
                for engine in ENGINES:
                    csv_path = exp_dir / f"{engine}_B{budget}.csv"
                    log(f"    [run] engine={engine} budget={budget} repr={repr_mode}")
                    if not args.dry_run:
                        run_solver(
                            project_root,
                            check_path,
                            [
                                "--engine",
                                engine,
                                "--stats",
                                "--repr",
                                repr_mode,
                                "--json",
                                rel_json,
                            ],
                            csv_path,
                        )
                    engine_csv[engine] = csv_path
                if args.dry_run:
                    continue
                line_objs = read_json_lines(json_path)
                pair_depths: List[float] = []
                pair_alt: List[float] = []
                pair_share: List[float] = []
                for line in line_objs:
                    meta = line.get("meta")
                    if meta:
                        u_meta = meta.get("u", {})
                        v_meta = meta.get("v", {})
                        pair_depths.append(float(max(u_meta.get("height", budget), v_meta.get("height", budget))))
                        pair_alt.append(float(meta.get("pair", {}).get("avg_alt_index", 0.0)))
                        pair_share.append(float(meta.get("pair", {}).get("avg_share_ratio", 0.0)))
                        shape_tag = (
                            line.get("config", {}).get("shape")
                            if exp.data["mode"] == "bag"
                            else exp.data.get("shape_label")
                        )
                        meta_rows.append(
                            {
                                "experiment": exp_id,
                                "budget": budget,
                                "shape_tag": shape_tag,
                                    "u_nodes": int(u_meta.get("nodes", 0)),
                                    "v_nodes": int(v_meta.get("nodes", 0)),
                                    "u_height": int(u_meta.get("height", 0)),
                                    "v_height": int(v_meta.get("height", 0)),
                                "avg_alt_index": float(meta.get("pair", {}).get("avg_alt_index", 0.0)),
                                "avg_share_ratio": float(meta.get("pair", {}).get("avg_share_ratio", 0.0)),
                            }
                        )
                    else:
                        pair_depths.append(float(budget))
                depth_q = quantiles(pair_depths)
                alt_q = quantiles(pair_alt)
                share_q = quantiles(pair_share)
                for engine in ENGINES:
                    rows = []
                    with engine_csv[engine].open("r", encoding="ascii") as handle:
                        reader = csv.DictReader(handle)
                        rows = list(reader)
                    totals = [float(row.get("total_us", 0.0)) for row in rows]
                    q = quantiles(totals)
                    avg = mean(totals)
                    for row in rows:
                        uv_stats = parse_stats_json(row.get("uv_stats", ""))
                        vu_stats = parse_stats_json(row.get("vu_stats", ""))
                        if uv_stats:
                            uv_entry = {
                                "experiment": exp_id,
                                "shape": exp.data["shape_label"],
                                "budget": budget,
                                "engine": engine,
                                "pair_id": int(row.get("pair_id", 0)),
                                "direction": "uv",
                            }
                            uv_entry.update(uv_stats)
                            stats_rows.append(uv_entry)
                        if vu_stats:
                            vu_entry = {
                                "experiment": exp_id,
                                "shape": exp.data["shape_label"],
                                "budget": budget,
                                "engine": engine,
                                "pair_id": int(row.get("pair_id", 0)),
                                "direction": "vu",
                            }
                            vu_entry.update(vu_stats)
                            stats_rows.append(vu_entry)
                    summary_rows.append(
                        {
                            "experiment": exp_id,
                            "shape": exp.data["shape_label"],
                            "budget": budget,
                            "median_pairDepth": depth_q["p50"],
                            "engine": engine,
                            "p25_us": q["p25"],
                            "median_us": q["p50"],
                            "p75_us": q["p75"],
                            "mean_us": round(avg, 3),
                            "n_pairs": len(rows),
                            "median_alt_index": round(alt_q["p50"], 6),
                            "median_share": round(share_q["p50"], 6),
                            "csv_path": engine_csv[engine].name,
                            "json_path": json_path.name,
                            "p_join": exp.data["p_join"],
                            "p_alt": exp.data["p_alt"],
                        }
                    )
            if args.dry_run:
                continue
            sum_path = exp_dir / "summary.csv"
            meta_path = exp_dir / "meta_pairs.csv"
            stats_path = exp_dir / "solver_stats.csv"
            stats_agg_path = exp_dir / "solver_stats_agg.csv"
            summary_rows.sort(key=lambda r: (r["engine"], r["budget"]))
            write_csv(sum_path, summary_rows)
            if meta_rows:
                write_csv(meta_path, meta_rows)
            if stats_rows:
                write_csv(stats_path, stats_rows)
                base_fields = {"experiment", "shape", "budget", "engine", "pair_id", "direction"}
                metric_names = set()
                for row in stats_rows:
                    metric_names.update(set(row.keys()) - base_fields)
                grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
                for row in stats_rows:
                    key = (
                        row["experiment"],
                        row["shape"],
                        row["budget"],
                        row["engine"],
                        row["direction"],
                    )
                        
                    grouped[key].append(row)
                agg_rows: List[Dict[str, Any]] = []
                for key, rows in grouped.items():
                    entry = {
                        "experiment": key[0],
                        "shape": key[1],
                        "budget": key[2],
                        "engine": key[3],
                        "direction": key[4],
                        "n": len(rows),
                    }
                    for metric in sorted(metric_names):
                        vals = []
                        for row in rows:
                            if metric in row:
                                try:
                                    vals.append(float(row[metric]))
                                except ValueError:
                                    continue
                        entry[f"avg_{metric}"] = mean(vals) if vals else 0.0
                    agg_rows.append(entry)
                agg_rows.sort(key=lambda r: (r["engine"], r["budget"], r["direction"]))
                if agg_rows:
                    write_csv(stats_agg_path, agg_rows)
            manifest_entry = {
                "id": exp_id,
                "representation": repr_mode,
                "mode": exp.data["mode"],
                "shape": exp.data["shape_label"],
                "budgets": ",".join(str(b) for b in budgets),
                "samples_requested": exp.data["samples"],
                "samples_used": samples_default,
                "vars": exp.data["vars"],
                "base_seed": args.stress_seed_base,
                "ensure_unique_leaves": exp.data.get("ensure_unique_leaves", False),
                "output_dir": os.path.relpath(exp_dir, project_root),
                "summary_csv": sum_path.name,
                "meta_csv": meta_path.name if meta_rows else "",
                "stats_csv": stats_path.name if stats_rows else "",
                "stats_agg_csv": stats_agg_path.name if stats_rows else "",
                "p_join": exp.data["p_join"],
                "p_alt": exp.data["p_alt"],
            }
            if exp.data["mode"] == "bag":
                manifest_entry["bag_shapes"] = ",".join(exp.data.get("bag_shapes", []))
            manifest_rows.append(manifest_entry)
        if args.dry_run:
            continue
        if manifest_rows:
            manifest_path = repr_out / "manifest.csv"
            write_csv(manifest_path, manifest_rows)
            log(f"[stress] manifest -> {manifest_path}")
    if args.dry_run:
        log("[stress] dry-run complete; no files written")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled+stress experiments without PowerShell")
    parser.add_argument("--repr", choices=["canonical", "legacy", "both"], default="both")
    parser.add_argument("--build", choices=["auto", "skip"], default="auto")
    parser.add_argument("--out-root")
    parser.add_argument("--budget-ceiling", type=int)
    parser.add_argument("--altfull-ceiling", type=int)
    parser.add_argument("--sample-override", type=int)
    parser.add_argument("--stress-sample-override", type=int, default=2)
    parser.add_argument("--filter")
    parser.add_argument("--skip-controlled", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=4294967296)
    parser.add_argument("--stress-seed-base", type=int, default=8589934592)
    parser.add_argument("--high-depth-threshold", type=int, default=12)
    parser.add_argument("--high-depth-cap", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    ensure_dir(project_root / BIN_DIR)
    build_binaries(project_root, args.build, args.dry_run)
    check_path = project_root / CHECK_EXE
    gen_path = project_root / GEN_EXE
    if not check_path.exists() or not gen_path.exists():
        raise RuntimeError("Missing required binaries after build step")
    representations = [args.repr] if args.repr in {"canonical", "legacy"} else ["canonical", "legacy"]
    filters = []
    if args.filter:
        filters = [token.strip() for token in args.filter.split(",") if token.strip()]
    if not args.skip_controlled:
        controlled_suite(project_root, args, representations, filters)
    else:
        log("Skipping controlled suite per configuration")
    if not args.skip_stress:
        stress_suite(project_root, args, representations, filters)
    else:
        log("Skipping stress suite per configuration")
    log("All requested suites finished successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
