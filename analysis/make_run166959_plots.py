"""Generate tree vs DAG comparison plots for run_166959.

This script creates visualizations comparing legacy (tree) vs canonical (DAG)
implementations across 4 lattice evaluation algorithms.

Plots generated:
1. Legacy runtime (log scale, depth 1-14)
2. Canonical runtime (log scale, depth 1-14)
3. Legacy memory (log scale, depth 1-14)
4. Canonical memory (log scale, depth 1-14)
5. Recursion comparison (Freese + Whitman, canonical vs legacy, depth 1-14)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE_STYLES = {
    "whitman": ("Whitman", "#d62728"),
    "freese": ("Freese", "#ff7f0e"),
    "cosma": ("Cosmadakis", "#1f77b4"),
    "hunt": ("Hunt", "#2ca02c"),
}

MEMORY_COLUMNS = (
    "uv_peak_working_set",
    "vu_peak_working_set",
)

MB = 1024 * 1024

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_summary(run_root: Path, mode: str, experiment: str) -> pd.DataFrame:
    """Load summary CSV for a specific mode (legacy/canonical) and experiment."""
    path = run_root / mode / experiment / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    
    df = pd.read_csv(path)
    
    # Rename engine column to engine_key for consistency
    if "engine" in df.columns:
        df["engine_key"] = df["engine"]
    
    # Rename budget to depth for consistency
    if "budget" in df.columns:
        df["depth"] = df["budget"]
    
    return df


def _subset_by_engine(df: pd.DataFrame, engine_key: str) -> pd.DataFrame:
    """Filter dataframe to single engine and sort by depth."""
    subset = df[df["engine_key"] == engine_key].copy()
    if "depth" in subset.columns:
        subset = subset.sort_values("depth")
    return subset


def _configure_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply standard axis configuration."""
    ax.set_xlabel("Depth", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_runtime_combined(
    legacy_summary: pd.DataFrame, 
    canonical_summary: pd.DataFrame, 
    out_dir: Path, 
    max_depth: int = 14
) -> None:
    """Create combined runtime plot: canonical (solid) + legacy (dashed) for ALL algorithms."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot canonical (DAG) as solid lines for ALL 4 engines
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(canonical_summary, engine_key)
        if subset.empty:
            continue
        
        # Filter to max_depth
        subset = subset[subset["depth"] <= max_depth]
        
        # Filter out zero values for log scale
        subset = subset[subset["median_us"] > 0]
        if subset.empty:
            continue
        
        # IQR ribbons if available
        if all(c in subset.columns for c in ["p25_us", "p75_us"]):
            ax.fill_between(
                subset["depth"].values,
                subset["p25_us"].values,
                subset["p75_us"].values,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1
            )
        
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=7,
            alpha=0.9,
            label=f"{engine_label} (DAG)",
            zorder=3
        )
    
    # Plot legacy (Tree) as dashed lines for ALL 4 engines
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(legacy_summary, engine_key)
        if subset.empty:
            continue
        
        # Filter to max_depth
        subset = subset[subset["depth"] <= max_depth]
        
        # Filter out zero values for log scale
        subset = subset[subset["median_us"] > 0]
        if subset.empty:
            continue
        
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.7,
            label=f"{engine_label} (Tree)",
            zorder=2
        )
    
    title = "Average Case: Tree vs DAG Runtime Performance (Depth 1-14)"
    ylabel = "Median Runtime (µs, log scale)"
    
    ax.set_yscale("log")
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95, ncol=2)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "runtime_tree_vs_dag.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved combined runtime plot")

def plot_recursion_tree_vs_dag(
    run_root: Path,
    legacy_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    out_dir: Path,
    max_depth: int = 14
) -> None:
    """Plot recursion comparison: Freese only, canonical vs legacy."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    experiment = "C1_altfull"
    
    # Plot both Whitman and Freese (recursive algorithms)
    for engine_key in ["whitman", "freese"]:
        if engine_key not in ENGINE_STYLES:
            continue
        
        engine_label, color = ENGINE_STYLES[engine_key]
        
        # Process both modes
        for mode, summary, linestyle, label_suffix in [
            ("legacy", legacy_summary, "--", " (Tree)"),
            ("canonical", canonical_summary, "-", " (DAG)"),
        ]:
            subset = _subset_by_engine(summary, engine_key)
            if subset.empty:
                continue
            
            recursion_data = []
            for _, row in subset.iterrows():
                if row["depth"] > max_depth:
                    continue
                
                csv_path = run_root / mode / experiment / row["csv_path"]
                if not csv_path.exists():
                    continue
                
                try:
                    per_pair = pd.read_csv(csv_path)
                    pair_totals = []
                    
                    # For each pair, sum recursive calls from both directions
                    for _, pair_row in per_pair.iterrows():
                        pair_total = 0
                        for col in ["uv_stats", "vu_stats"]:
                            if col in pair_row:
                                try:
                                    stats = json.loads(pair_row[col])
                                    # Whitman uses "pairs", others use "recursive_calls"
                                    if "recursive_calls" in stats:
                                        pair_total += stats["recursive_calls"]
                                    elif "pairs" in stats:
                                        pair_total += stats["pairs"]
                                except (json.JSONDecodeError, TypeError):
                                    continue
                        if pair_total > 0:
                            pair_totals.append(pair_total)
                    
                    if pair_totals:
                        recursion_data.append((row["depth"], np.median(pair_totals)))
                
                except Exception:
                    continue
            
            if not recursion_data:
                continue
            
            recursion_data.sort(key=lambda x: x[0])
            depths = [item[0] for item in recursion_data]
            counts = [item[1] for item in recursion_data]
            
            ax.plot(
                depths,
                counts,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker="o",
                markersize=6,
                alpha=0.9,
                label=f"{engine_label}{label_suffix}",
            )
    
    title = "Recursive Work: Tree vs DAG (Depth 1-14)"
    ylabel = "Median Recursive Calls (log scale)"
    ax.set_yscale("log")
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95, ncol=2)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "recursion_tree_vs_dag.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved recursion tree vs DAG plot")


def plot_speedup_factor(
    legacy_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    out_dir: Path,
    max_depth: int = 14
) -> None:
    """Plot speedup factor: Tree runtime / DAG runtime for ALL algorithms."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot all 4 algorithms
    for engine_key in ["whitman", "freese", "cosma", "hunt"]:
        if engine_key not in ENGINE_STYLES:
            continue
        
        engine_label, color = ENGINE_STYLES[engine_key]
        
        # Get tree and DAG data
        tree_data = _subset_by_engine(legacy_summary, engine_key)
        dag_data = _subset_by_engine(canonical_summary, engine_key)
        
        if tree_data.empty or dag_data.empty:
            continue
        
        # Filter to max_depth
        tree_data = tree_data[tree_data["depth"] <= max_depth].sort_values("depth")
        dag_data = dag_data[dag_data["depth"] <= max_depth].sort_values("depth")
        
        # Merge on depth to compute speedup
        merged = tree_data[["depth", "median_us"]].merge(
            dag_data[["depth", "median_us"]], 
            on="depth", 
            suffixes=("_tree", "_dag")
        )
        
        # Filter out zeros to avoid division by zero or invalid speedup
        merged = merged[(merged["median_us_tree"] > 0) & (merged["median_us_dag"] > 0)]
        if merged.empty:
            continue
        
        # Compute speedup factor
        merged["speedup"] = merged["median_us_tree"] / merged["median_us_dag"]
        
        # Plot
        ax.plot(
            merged["depth"],
            merged["speedup"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=7,
            alpha=0.9,
            label=engine_label,
            zorder=3
        )
    
    # Add horizontal line at y=1.0 (no speedup reference)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.text(max_depth * 0.98, 1.05, 'No speedup', ha='right', va='bottom', 
            fontsize=10, color='gray', alpha=0.8)
    
    title = "Average Case: DAG Speedup Over Tree"
    ylabel = "Speedup Factor (Tree runtime / DAG runtime)"
    
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    ax.set_ylim(bottom=0.1)  # Allow for Freese's low values at shallow depths
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "speedup_factor.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved speedup factor plot")


def plot_case_runtime_combined(
    legacy_summary: pd.DataFrame, 
    canonical_summary: pd.DataFrame, 
    out_dir: Path,
    case_name: str,
    filename: str
) -> None:
    """Create combined runtime plot for a specific case: canonical (solid) + legacy (dashed)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot canonical (DAG) as solid lines for all engines
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(canonical_summary, engine_key)
        if subset.empty:
            continue
        
        subset = subset.sort_values("depth")
        
        # Filter out zero values for log scale
        subset = subset[subset["median_us"] > 0]
        if subset.empty:
            continue
        
        # IQR ribbons if available
        if all(c in subset.columns for c in ["p25_us", "p75_us"]):
            ax.fill_between(
                subset["depth"].values,
                subset["p25_us"].values,
                subset["p75_us"].values,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1
            )
        
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=7,
            alpha=0.9,
            label=f"{engine_label} (DAG)",
            zorder=3
        )
    
    # Plot legacy (Tree) as dashed lines for ALL 4 engines
    for engine_key in ["whitman", "freese", "cosma", "hunt"]:
        if engine_key not in ENGINE_STYLES:
            continue
        engine_label, color = ENGINE_STYLES[engine_key]
        
        subset = _subset_by_engine(legacy_summary, engine_key)
        if subset.empty:
            continue
        
        subset = subset.sort_values("depth")
        
        # Filter out zero values for log scale
        subset = subset[subset["median_us"] > 0]
        if subset.empty:
            continue
        
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.7,
            label=f"{engine_label} (Tree)",
            zorder=2
        )
    
    title = f"{case_name}: Tree vs DAG Runtime Performance"
    ylabel = "Median Runtime (µs, log scale)"
    
    ax.set_yscale("log")
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95, ncol=2)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved {case_name.lower()} combined runtime plot")


def plot_case_speedup_factor(
    legacy_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    out_dir: Path,
    case_name: str,
    filename: str
) -> None:
    """Plot speedup factor for a specific case: Tree runtime / DAG runtime."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Loop through ALL 4 engines
    for engine_key in ["whitman", "freese", "cosma", "hunt"]:
        if engine_key not in ENGINE_STYLES:
            continue
        
        engine_label, color = ENGINE_STYLES[engine_key]
        
        # Get tree and DAG data
        tree_data = _subset_by_engine(legacy_summary, engine_key)
        dag_data = _subset_by_engine(canonical_summary, engine_key)
        
        if tree_data.empty or dag_data.empty:
            continue
        
        tree_data = tree_data.sort_values("depth")
        dag_data = dag_data.sort_values("depth")
        
        # Merge on depth to compute speedup
        merged = tree_data[["depth", "median_us"]].merge(
            dag_data[["depth", "median_us"]], 
            on="depth", 
            suffixes=("_tree", "_dag")
        )
        
        # Filter out zeros to avoid division by zero or invalid speedup
        merged = merged[(merged["median_us_tree"] > 0) & (merged["median_us_dag"] > 0)]
        if merged.empty:
            continue
        
        # Compute speedup factor
        merged["speedup"] = merged["median_us_tree"] / merged["median_us_dag"]
        
        # Plot
        ax.plot(
            merged["depth"],
            merged["speedup"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=7,
            alpha=0.9,
            label=engine_label,
            zorder=3
        )
    
    # Add horizontal line at y=1.0 (no speedup reference)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    max_depth = max(legacy_summary["depth"].max(), canonical_summary["depth"].max()) if not legacy_summary.empty else 14
    ax.text(max_depth * 0.98, 1.05, 'No speedup', ha='right', va='bottom', 
            fontsize=10, color='gray', alpha=0.8)
    
    title = f"{case_name}: DAG Speedup Over Tree"
    ylabel = "Speedup Factor (Tree runtime / DAG runtime)"
    
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    ax.set_ylim(bottom=0.5)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved {case_name.lower()} speedup factor plot")


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------

def generate_plots(run_root: Path, out_dir: Path) -> None:
    """Generate all tree vs DAG comparison plots."""
    print("Loading data...")
    
    # Load summaries for both modes and all three experiments
    legacy_avg = load_summary(run_root, "legacy", "C1_altfull")
    canonical_avg = load_summary(run_root, "canonical", "C1_altfull")
    legacy_best = load_summary(run_root, "legacy", "E_bestcase")
    canonical_best = load_summary(run_root, "canonical", "E_bestcase")
    legacy_worst = load_summary(run_root, "legacy", "E_worstcase")
    canonical_worst = load_summary(run_root, "canonical", "E_worstcase")
    
    print("\n=== AVERAGE CASE (C1_altfull, depth 1-14) ===")
    
    print("\nGenerating combined runtime plot...")
    plot_runtime_combined(legacy_avg, canonical_avg, out_dir, max_depth=14)
    
    print("\nGenerating recursion comparison plot...")
    plot_recursion_tree_vs_dag(run_root, legacy_avg, canonical_avg, out_dir, max_depth=14)
    
    print("\nGenerating speedup factor plot...")
    plot_speedup_factor(legacy_avg, canonical_avg, out_dir, max_depth=14)
    
    print("\n=== BEST CASE (E_bestcase, depth 1-14) ===")
    
    print("\nGenerating best-case combined runtime plot...")
    plot_case_runtime_combined(legacy_best, canonical_best, out_dir, 
                               "Best Case", "bestcase_runtime_tree_vs_dag.png")
    
    print("\nGenerating best-case speedup factor plot...")
    plot_case_speedup_factor(legacy_best, canonical_best, out_dir,
                            "Best Case", "bestcase_speedup_factor.png")
    
    print("\n=== WORST CASE (E_worstcase, depth 1-14) ===")
    
    print("\nGenerating worst-case combined runtime plot...")
    plot_case_runtime_combined(legacy_worst, canonical_worst, out_dir,
                              "Worst Case", "worstcase_runtime_tree_vs_dag.png")
    
    print("\nGenerating worst-case speedup factor plot...")
    plot_case_speedup_factor(legacy_worst, canonical_worst, out_dir,
                            "Worst Case", "worstcase_speedup_factor.png")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {out_dir}")
    print(f"{'='*60}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tree vs DAG comparison plots for run_166959"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/run_166959"),
        help="Root directory containing legacy and canonical results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/plots_run_166959"),
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    run_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    
    if not run_root.exists():
        raise SystemExit(f"Results directory not found: {run_root}")
    
    generate_plots(run_root, out_dir)


if __name__ == "__main__":
    main()
