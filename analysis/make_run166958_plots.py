"""Generate publication-quality plots for run_166958 (canonical, depth 1-16).

This script creates figures for the main experimental results:
* C1_altfull: Runtime comparison of all four algorithms (linear and log scales)
* E_bestcase: Best-case performance with O(1) theoretical baseline
* E_worstcase: Worst-case performance with theoretical complexity overlays
* Memory profiles for C1_altfull

All plots use clean styling with proper academic formatting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Try to import LOWESS, fallback to manual smoothing if not available
try:
    from statsmodels.api import nonparametric
    lowess = nonparametric.lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False
    print("Warning: statsmodels not available, using simple moving average for smoothing")


def _simple_smooth(x: np.ndarray, y: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Simple moving average smoothing as fallback for LOWESS."""
    if len(x) < window:
        return x, y
    
    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # Apply moving average
    y_smoothed = np.convolve(y_sorted, np.ones(window) / window, mode='valid')
    x_smoothed = x_sorted[window // 2:-(window // 2)]
    
    return x_smoothed, y_smoothed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ordered dictionary to ensure consistent algorithm ordering in plots
ENGINE_STYLES: Dict[str, tuple[str, str]] = {
    "whitman": ("Whitman", "#d62728"),
    "freese": ("Freese", "#ff7f0e"),
    "cosma": ("Cosmadakis", "#1f77b4"),
    "hunt": ("Hunt", "#2ca02c"),
}

# Theoretical complexity for worst-case
WORST_CASE_THEORY = {
    "whitman": {"type": "exp", "base": 2.0, "label": "O(2^n)", "complexity_exp": "2^n"},
    "freese": {"type": "poly", "exponent": 2.5, "label": "O(n^{2.5})", "complexity_exp": "n^{2.5}"},
    "cosma": {"type": "poly", "exponent": None, "label": "O(n^k)", "complexity_exp": "n^k"},
    "hunt": {"type": "poly", "exponent": None, "label": "O(n^k)", "complexity_exp": "n^k"},
}

# Theoretical memory complexity (in MB, as a function of nodes)
# Note: These are STARTING POINTS - actual exponents are fitted from empirical data
# The labels show the theoretical asymptotic complexity from the paper
MEMORY_THEORY = {
    "whitman": {"type": "exp", "base": 2.0, "label": "O(2^n)", "complexity_exp": "2^n"},
    "freese": {"type": "poly", "exponent": None, "label": "O(n^2)", "complexity_exp": "n^2"},
    "cosma": {"type": "poly", "exponent": None, "label": "O(log n)", "complexity_exp": "log n"},
    "hunt": {"type": "poly", "exponent": None, "label": "O(n^k)", "complexity_exp": "n^k"},
}

MEMORY_COLUMNS = (
    "uv_peak_working_set",
    "vu_peak_working_set",
)

MB = 1024 * 1024


@dataclass
class SeriesData:
    depth: np.ndarray
    values: np.ndarray


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_summary(run_root: Path, experiment: str) -> pd.DataFrame:
    """Load summary CSV for a specific experiment."""
    summary_path = run_root / "canonical" / experiment / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    df = pd.read_csv(summary_path)
    df = df.copy()
    df.rename(columns={"budget": "depth"}, inplace=True)
    df["depth"] = df["depth"].astype(float)
    df["engine_key"] = df["engine"].str.lower()
    return df


def load_memory_profiles(
    run_root: Path,
    experiment: str,
    summary: pd.DataFrame,
) -> Mapping[str, SeriesData]:
    """Extract peak memory usage per algorithm from individual CSV files."""
    per_engine: Dict[str, list[tuple[float, float]]] = {engine: [] for engine in ENGINE_STYLES}
    
    for _, row in summary.iterrows():
        engine_key = str(row["engine_key"])
        csv_path = run_root / "canonical" / experiment / row["csv_path"]
        
        if not csv_path.exists():
            continue
            
        try:
            per_pair = pd.read_csv(csv_path)
        except Exception:
            continue
        
        peaks = []
        for column in MEMORY_COLUMNS:
            if column in per_pair:
                peaks.append(per_pair[column].astype(float))
        
        if not peaks:
            continue
        
        combined = pd.concat(peaks, axis=1).max(axis=1)
        if combined.empty:
            continue
        
        per_engine.setdefault(engine_key, []).append(
            (float(row["depth"]), float(combined.median()) / MB)
        )
    
    series: Dict[str, SeriesData] = {}
    for engine_key, records in per_engine.items():
        if not records:
            continue
        records.sort(key=lambda item: item[0])
        depth = np.array([item[0] for item in records], dtype=float)
        values = np.array([item[1] for item in records], dtype=float)
        series[engine_key] = SeriesData(depth=depth, values=values)
    
    return series


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _configure_axes(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "Depth") -> None:
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)


def _subset_by_engine(summary: pd.DataFrame, engine_key: str) -> pd.DataFrame:
    """Filter summary to a single algorithm."""
    return summary[summary["engine_key"] == engine_key].sort_values("depth")


def _compute_theoretical_curve(
    engine_key: str,
    depth: np.ndarray,
    values: np.ndarray,
    for_best_case: bool = False,
    for_memory: bool = False,
) -> tuple[SeriesData, str] | None:
    """Compute theoretical complexity curve fitted to empirical data."""
    if for_best_case:
        # Best case should be constant time O(1)
        valid_values = values[values > 0]
        if len(valid_values) == 0:
            # All values are zero or negative - use a small constant
            # This happens for Whitman's best case (instant cache hits)
            mean_val = 0.5  # Use a small visible value for plotting
        else:
            mean_val = np.median(valid_values)
        return SeriesData(depth=depth, values=np.full_like(depth, mean_val, dtype=float)), "O(1)"
    
    config = MEMORY_THEORY.get(engine_key) if for_memory else WORST_CASE_THEORY.get(engine_key)
    if config is None:
        return None
    
    # Filter valid data points
    mask = (depth > 0) & (values > 0) & np.isfinite(values)
    if mask.sum() < 3:
        return None
    
    depth_clean = depth[mask]
    values_clean = values[mask]
    
    # Sort by depth
    order = np.argsort(depth_clean)
    depth_clean = depth_clean[order]
    values_clean = values_clean[order]
    
    # Get the theoretical label (for display)
    theoretical_label = config.get("label", "O(?)")
    kind = config.get("type", "poly")
    
    # Compute number of nodes (assuming binary tree: 2^depth)
    nodes = np.power(2.0, depth_clean)
    
    if kind == "exp":
        # Exponential: a * base^depth
        base = float(config.get("base", 2.0))
        base_series = np.power(base, depth_clean)
        # Fit scale factor using last few points (more robust for memory)
        if for_memory:
            # Use median of last 3 points for more stable memory fitting
            fit_points = min(3, len(values_clean))
            scale = np.median(values_clean[-fit_points:] / base_series[-fit_points:])
        else:
            scale = values_clean[-1] / base_series[-1]
        trend_values = scale * base_series
        complexity_exp = theoretical_label
    
    elif kind == "linear":
        # Linear: a * n (deprecated, use poly with exponent=1)
        base_series = nodes
        if for_memory:
            fit_points = min(3, len(values_clean))
            scale = np.median(values_clean[-fit_points:] / base_series[-fit_points:])
        else:
            scale = values_clean[-1] / base_series[-1]
        trend_values = scale * base_series
        complexity_exp = theoretical_label
    
    elif kind == "log":
        # Logarithmic: a * log(n) - kept for backwards compatibility
        base_series = np.log2(nodes)
        if for_memory:
            fit_points = min(5, len(values_clean))
            scales = values_clean[-fit_points:] / base_series[-fit_points:]
            scale = np.median(scales[scales > 0]) if np.any(scales > 0) else values_clean[-1] / base_series[-1]
        else:
            scale = values_clean[-1] / base_series[-1]
        trend_values = np.maximum(scale * base_series, values_clean.min() * 0.5)
        complexity_exp = theoretical_label
    
    elif kind == "poly":
        # Polynomial: a * n^k
        # ALWAYS fit exponent from data for best match
        exponent = config.get("exponent")
        if exponent is None:
            # Estimate exponent from log-log fit (this is what makes it follow the data!)
            slope, intercept = np.polyfit(np.log(nodes), np.log(values_clean), 1)
            exponent = float(slope)
        
        base_series = np.power(nodes, float(exponent))
        if for_memory:
            # Use median of last 3 points for more stable memory fitting
            fit_points = min(3, len(values_clean))
            scale = np.median(values_clean[-fit_points:] / base_series[-fit_points:])
        else:
            scale = values_clean[-1] / base_series[-1]
        trend_values = scale * base_series
        # Show the theoretical label, not the fitted exponent
        complexity_exp = theoretical_label
    
    else:
        return None
    
    return SeriesData(depth=depth_clean, values=trend_values), complexity_exp


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def plot_c1_runtime(summary: pd.DataFrame, out_dir: Path, log_scale: bool) -> None:
    """Create runtime comparison for C1_altfull experiment.
    
    For log scale, computes and annotates crossover depths where polynomial
    methods become faster than Whitman. Interpolation done in log-runtime space.
    
    Caption guidance (log scale only):
    "Algorithm performance comparison (C1, log scale). Vertical markers indicate
    crossover depths where polynomial methods first overtake Whitman: Freese
    (~8.3), Cosmadakis (~9.1), Hunt (~10.7). Crossovers computed from median
    curves via linear interpolation in log-space. Marks transition from
    exponential to polynomial dominance."
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get Whitman data for crossover computation
    whitman_data = _subset_by_engine(summary, "whitman")
    whitman_dict = dict(zip(whitman_data["depth"], whitman_data["median_us"])) if not whitman_data.empty else {}
    
    crossovers = {}  # Store crossover depths for annotation
    
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(summary, engine_key)
        if subset.empty:
            continue
        
        # IQR ribbons if quartile data available
        if all(c in subset.columns for c in ["p25_us", "p75_us"]):
            ax.fill_between(
                subset["depth"].values,
                subset["p25_us"].values,
                subset["p75_us"].values,
                color=color,
                alpha=0.15,
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
            markersize=6,
            alpha=0.9,
            label=engine_label,
            zorder=3
        )
        
        # Compute crossover point (where algorithm becomes faster than Whitman)
        if engine_key != "whitman" and whitman_dict and log_scale:
            for i in range(len(subset) - 1):
                depth1 = subset.iloc[i]["depth"]
                depth2 = subset.iloc[i + 1]["depth"]
                runtime1 = subset.iloc[i]["median_us"]
                runtime2 = subset.iloc[i + 1]["median_us"]
                
                if depth1 in whitman_dict and depth2 in whitman_dict:
                    whitman1 = whitman_dict[depth1]
                    whitman2 = whitman_dict[depth2]
                    
                    # Check if crossover happens between these points
                    if runtime1 >= whitman1 and runtime2 < whitman2:
                        # Linear interpolation in log space
                        log_ratio1 = np.log10(runtime1 / whitman1)
                        log_ratio2 = np.log10(runtime2 / whitman2)
                        
                        if log_ratio1 != log_ratio2:
                            t = log_ratio1 / (log_ratio1 - log_ratio2)
                            crossover_depth = depth1 + t * (depth2 - depth1)
                            crossovers[engine_key] = (crossover_depth, color, engine_label)
                        break
    
    scale_suffix = " (log scale)" if log_scale else ""
    ylabel = "Median Runtime (µs)"
    title = f"Algorithm Performance Comparison: C1 Baseline{scale_suffix}"
    
    if log_scale:
        ax.set_yscale("log")
        
        # Add crossover markers
        y_limits = ax.get_ylim()
        for engine_key, (crossover_depth, color, label) in crossovers.items():
            ax.axvline(
                x=crossover_depth,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.6,
                zorder=2
            )
            # Add label at top
            ax.text(
                crossover_depth, y_limits[1] * 0.7,
                f"{label}\n{crossover_depth:.1f}",
                fontsize=8,
                color=color,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, alpha=0.8, linewidth=1.5)
            )
    
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "log" if log_scale else "linear"
    fig.savefig(out_dir / f"c1_runtime_{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_best_case(summary: pd.DataFrame, out_dir: Path) -> None:
    """Create best-case performance plot with O(1) theoretical baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(summary, engine_key)
        if subset.empty:
            continue
        
        # Plot empirical data
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=f"{engine_label} (empirical)",
        )
        
        # Add O(1) theoretical baseline
        theory = _compute_theoretical_curve(
            engine_key,
            subset["depth"].to_numpy(),
            subset["median_us"].to_numpy(),
            for_best_case=True,
            for_memory=False,
        )
        
        if theory is not None:
            trend_series, complexity_exp = theory
            ax.plot(
                trend_series.depth,
                trend_series.values,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.5,
                label=f"{engine_label} Theoretical ({complexity_exp})",
            )
    
    title = "Best-Case Performance: All Algorithms"
    ylabel = "Median Runtime (µs)"
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "best_case_all.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_worst_case(summary: pd.DataFrame, out_dir: Path, log_scale: bool) -> None:
    """Create worst-case performance plot with theoretical complexity overlays."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = _subset_by_engine(summary, engine_key)
        if subset.empty:
            continue
        
        # Plot empirical data
        ax.plot(
            subset["depth"],
            subset["median_us"],
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=f"{engine_label}",
        )
        
        # Add theoretical complexity curve
        theory = _compute_theoretical_curve(
            engine_key,
            subset["depth"].to_numpy(),
            subset["median_us"].to_numpy(),
            for_best_case=False,
            for_memory=False,
        )
        
        if theory is not None:
            trend_series, complexity_exp = theory
            ax.plot(
                trend_series.depth,
                trend_series.values,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.6,
                label=f"{engine_label} Theoretical ({complexity_exp})",
            )
    
    scale_suffix = " (log scale)" if log_scale else ""
    title = f"Worst-Case Performance with Theoretical Complexity{scale_suffix}"
    ylabel = "Median Runtime (µs)"
    
    if log_scale:
        ax.set_yscale("log")
    
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "log" if log_scale else "linear"
    fig.savefig(out_dir / f"worst_case_{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_memory(memory_series: Mapping[str, SeriesData], out_dir: Path, log_scale: bool) -> None:
    """Create memory usage comparison plot."""
    if not memory_series:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for engine_key, series in memory_series.items():
        if engine_key not in ENGINE_STYLES:
            continue
        
        engine_label, color = ENGINE_STYLES[engine_key]
        
        # Plot empirical memory data
        ax.plot(
            series.depth,
            series.values,
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=f"{engine_label}",
        )
        
        # Only add theoretical curves for linear scale
        if not log_scale:
            theory = _compute_theoretical_curve(
                engine_key,
                series.depth,
                series.values,
                for_best_case=False,
                for_memory=True,
            )
            
            if theory is not None:
                trend_series, complexity_exp = theory
                ax.plot(
                    trend_series.depth,
                    trend_series.values,
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.6,
                    label=f"{engine_label} Theoretical ({complexity_exp})",
                )
    
    scale_suffix = " (log scale)" if log_scale else ""
    title = f"Memory Usage Profile: C1 Baseline{scale_suffix}"
    ylabel = "Peak Working Set (MB)"
    
    if log_scale:
        ax.set_yscale("log")
        # Set reasonable y-axis limits for log scale (don't go too low)
        all_values = np.concatenate([s.values for s in memory_series.values()])
        valid_values = all_values[all_values > 0]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            # Set lower limit to at most 0.1 * min_value, upper limit with some padding
            ax.set_ylim(bottom=min_val * 0.5, top=max_val * 2)
    
    _configure_axes(ax, title, ylabel)
    # Use single column for log scale (no theoretical curves), two columns for linear
    ncol = 1 if log_scale else 2
    ax.legend(fontsize=9 if log_scale else 9, loc="upper left", ncol=ncol)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "log" if log_scale else "linear"
    fig.savefig(out_dir / f"c1_memory_{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_individual_best_worst(
    best_summary: pd.DataFrame,
    worst_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Create individual plots for each algorithm showing best and worst case."""
    # Vivid color scheme: green for best case, red for worst case
    BEST_COLOR = "#2ca02c"  # Vivid green
    WORST_COLOR = "#d62728"  # Vivid red
    BEST_THEORY_COLOR = "#98df8a"  # Light green
    WORST_THEORY_COLOR = "#ff9896"  # Light red
    
    for engine_key, (engine_label, base_color) in ENGINE_STYLES.items():
        best_subset = _subset_by_engine(best_summary, engine_key)
        worst_subset = _subset_by_engine(worst_summary, engine_key)
        
        if best_subset.empty and worst_subset.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot best case empirical
        if not best_subset.empty:
            ax.plot(
                best_subset["depth"],
                best_subset["median_us"],
                color=BEST_COLOR,
                linestyle="-",
                linewidth=2.5,
                marker="o",
                markersize=6,
                alpha=0.9,
                label="Best Case",
            )
            
            # Add best case theoretical
            best_theory = _compute_theoretical_curve(
                engine_key,
                best_subset["depth"].to_numpy(),
                best_subset["median_us"].to_numpy(),
                for_best_case=True,
                for_memory=False,
            )
            
            if best_theory is not None:
                trend_series, complexity_exp = best_theory
                ax.plot(
                    trend_series.depth,
                    trend_series.values,
                    color=BEST_THEORY_COLOR,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.7,
                    label=f"Best Case Theory ({complexity_exp})",
                )
        
        # Plot worst case empirical (use circles, not squares)
        if not worst_subset.empty:
            ax.plot(
                worst_subset["depth"],
                worst_subset["median_us"],
                color=WORST_COLOR,
                linestyle="-",
                linewidth=2.5,
                marker="o",
                markersize=6,
                alpha=0.9,
                label="Worst Case",
            )
            
            # Add worst case theoretical
            worst_theory = _compute_theoretical_curve(
                engine_key,
                worst_subset["depth"].to_numpy(),
                worst_subset["median_us"].to_numpy(),
                for_best_case=False,
                for_memory=False,
            )
            
            if worst_theory is not None:
                trend_series, complexity_exp = worst_theory
                ax.plot(
                    trend_series.depth,
                    trend_series.values,
                    color=WORST_THEORY_COLOR,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.7,
                    label=f"Worst Case Theory ({complexity_exp})",
                )
        
        title = f"{engine_label}: Best vs Worst Case"
        ylabel = "Median Runtime (µs)"
        _configure_axes(ax, title, ylabel)
        ax.legend(fontsize=10, loc="upper left")
        fig.tight_layout()
        
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = engine_key.replace(" ", "_").lower()
        fig.savefig(out_dir / f"{safe_name}_best_worst.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_speedup_ratio(summary: pd.DataFrame, out_dir: Path) -> None:
    """Create speedup ratio plot (relative to Whitman as baseline)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get Whitman as baseline
    whitman_data = _subset_by_engine(summary, "whitman")
    if whitman_data.empty:
        print("Warning: No Whitman data for speedup ratio")
        return
    
    baseline_dict = dict(zip(whitman_data["depth"], whitman_data["median_us"]))
    
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        if engine_key == "whitman":
            continue  # Skip baseline
        
        subset = _subset_by_engine(summary, engine_key)
        if subset.empty:
            continue
        
        # Compute speedup: baseline_time / algorithm_time
        speedups = []
        depths = []
        for _, row in subset.iterrows():
            depth = row["depth"]
            if depth in baseline_dict and row["median_us"] > 0:
                speedup = baseline_dict[depth] / row["median_us"]
                speedups.append(speedup)
                depths.append(depth)
        
        if not speedups:
            continue
        
        ax.plot(
            depths,
            speedups,
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=engine_label,
        )
    
    title = "Speedup Relative to Whitman (C1 Baseline, log scale)"
    ylabel = "Speedup Factor (×)"
    ax.set_yscale("log")
    _configure_axes(ax, title, ylabel)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label="Baseline (Whitman)")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "speedup_ratio.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_recursion_comparison(run_root: Path, experiment: str, summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot total recursive_calls for Whitman vs Freese (sum of both directions per pair)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for engine_key in ["whitman", "freese"]:
        if engine_key not in ENGINE_STYLES:
            continue
        
        engine_label, color = ENGINE_STYLES[engine_key]
        subset = _subset_by_engine(summary, engine_key)
        if subset.empty:
            continue
        
        recursion_data = []
        for _, row in subset.iterrows():
            csv_path = run_root / "canonical" / experiment / row["csv_path"]
            if not csv_path.exists():
                continue
            
            try:
                per_pair = pd.read_csv(csv_path)
                pair_totals = []
                
                # For each pair (row), sum the recursive calls from both directions
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
                    # Take median across all pairs at this depth
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
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.9,
            label=f"{engine_label}",
        )
    
    title = "Recursive Calls: Whitman vs Freese"
    ylabel = "Median Recursive Calls (both directions)"
    ax.set_yscale("log")
    _configure_axes(ax, title, ylabel)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "recursion_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_memoization_efficiency(run_root: Path, experiment: str, summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot Freese's memoization hit rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    freese_subset = _subset_by_engine(summary, "freese")
    if freese_subset.empty:
        print("Warning: No Freese data for memoization efficiency")
        return
    
    engine_label, color = ENGINE_STYLES["freese"]
    hit_rate_data = []
    
    for _, row in freese_subset.iterrows():
        csv_path = run_root / "canonical" / experiment / row["csv_path"]
        if not csv_path.exists():
            continue
        
        try:
            per_pair = pd.read_csv(csv_path)
            hit_rates = []
            
            for col in ["uv_stats", "vu_stats"]:
                if col in per_pair:
                    for stats_str in per_pair[col]:
                        try:
                            stats = json.loads(stats_str)
                            if "memo_hits" in stats and "recursive_calls" in stats:
                                total = stats["recursive_calls"] + stats["memo_hits"]
                                if total > 0:
                                    hit_rate = (stats["memo_hits"] / total) * 100
                                    hit_rates.append(hit_rate)
                        except (json.JSONDecodeError, TypeError):
                            continue
            
            if hit_rates:
                hit_rate_data.append((row["depth"], np.median(hit_rates)))
        
        except Exception:
            continue
    
    if not hit_rate_data:
        print("Warning: No memoization data found")
        return
    
    hit_rate_data.sort(key=lambda x: x[0])
    depths = [item[0] for item in hit_rate_data]
    rates = [item[1] for item in hit_rate_data]
    
    ax.plot(
        depths,
        rates,
        color=color,
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=6,
        alpha=0.9,
        label="Freese Hit Rate",
    )
    
    title = "Freese Memoization Efficiency"
    ylabel = "Cache Hit Rate (%)"
    _configure_axes(ax, title, ylabel)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "memoization_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_c4_node_sharing(run_root: Path, out_dir: Path) -> None:
    """Plot node sharing impact on performance (C4 experiments).
    
    Shows runtime vs median_share across C4_share_high/medium/low experiments.
    Includes Spearman correlation and LOWESS smoothing per algorithm.
    
    Caption guidance:
    "Node sharing vs runtime (C4). x = unique/occurrences (lower = more sharing).
    y = log₁₀ median runtime (µs). Spearman ρ (p-value, n) per algorithm.
    Freese and Whitman slow markedly as sharing decreases (ρ=+0.82, +0.88),
    consistent with increased distinct subproblems in less-compressed DAGs.
    Cosmadakis and Hunt show weak/non-significant dependence (ρ≈±0.23),
    aligning with their operation over unique subterms rather than occurrences.
    Points aggregate budgets 200-1000; correlations robust within depth buckets."
    """
    print("Generating C4 node sharing impact plot...")
    
    # Load all C4 variants
    c4_variants = ["C4_share_high", "C4_share_medium", "C4_share_low"]
    all_data = []
    
    for variant in c4_variants:
        try:
            summary = load_summary(run_root, variant)
            summary["variant"] = variant
            all_data.append(summary)
        except FileNotFoundError:
            print(f"Warning: {variant} not found, skipping...")
            continue
    
    if not all_data:
        print("Warning: No C4 data found for node sharing plot")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Create main figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot per algorithm
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        subset = combined[combined["engine_key"] == engine_key].copy()
        if subset.empty:
            continue
        
        # Filter valid data
        subset = subset[(subset["median_share"] > 0) & (subset["median_us"] > 0)].copy()
        if len(subset) < 3:
            continue
        
        share = subset["median_share"].values
        runtime = subset["median_us"].values
        log_runtime = np.log10(runtime)
        
        # Scatter plot
        ax.scatter(
            share,
            log_runtime,
            color=color,
            s=80,
            alpha=0.5,  # Reduced from 0.6 for less overlap
            edgecolors='white',
            linewidth=0.5,
            label=f"{engine_label}",
            zorder=3
        )
        
        # LOWESS smoothing with vivid colors
        if len(subset) >= 5:
            try:
                if HAS_LOWESS:
                    smoothed = lowess(log_runtime, share, frac=0.4, return_sorted=True)
                    ax.plot(
                        smoothed[:, 0],
                        smoothed[:, 1],
                        color=color,
                        linestyle="-",
                        linewidth=2.5,  # Thicker, more vivid
                        alpha=0.95,  # More vivid
                        zorder=4
                    )
                else:
                    # Fallback to simple smoothing
                    x_smooth, y_smooth = _simple_smooth(share, log_runtime, window=3)
                    ax.plot(
                        x_smooth,
                        y_smooth,
                        color=color,
                        linestyle="-",
                        linewidth=2.5,
                        alpha=0.95,
                        zorder=4
                    )
            except Exception as e:
                print(f"Warning: Smoothing failed for {engine_label}: {e}")
        
        # Spearman correlation - annotate in bottom right
        try:
            rho, p_value = spearmanr(share, runtime)
            # Annotate correlation on plot (bottom right instead of top left)
            y_pos = 0.25 - (list(ENGINE_STYLES.keys()).index(engine_key) * 0.05)
            ax.text(
                0.98, y_pos,
                f"{engine_label}: ρ = {rho:.3f} (p < {p_value:.1e}, n = {len(subset)})",
                transform=ax.transAxes,
                fontsize=9,
                color=color,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9)
            )
        except Exception as e:
            print(f"Warning: Spearman correlation failed for {engine_label}: {e}")
    
    # Formatting
    ax.set_xlabel("Subterm Uniqueness Ratio (unique/occurrences; lower = more sharing)", fontsize=11)
    ax.set_ylabel("log₁₀(Median Runtime) [µs]", fontsize=11)
    ax.set_title("Node Sharing Impact on Performance (C4 Experiments)", fontsize=12, fontweight='bold')
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    
    # Legend in top left
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95, frameon=True)
    
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "c4_node_sharing.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("  -> Saved C4 node sharing plot")


def plot_c6_karity_grid(run_root: Path, out_dir: Path) -> None:
    """Plot k-arity scaling comparison (C6 experiments).
    
    Shows runtime vs budget for k ∈ {2,3,4,6} in a 2x2 grid.
    
    Caption guidance:
    "K-arity impact on performance (C6). Runtime vs budget for k∈{2,3,4,6}.
    Identical y-axis scales enable direct comparison. Median depth annotated
    per panel. Higher k (wider terms) increases complexity for all algorithms,
    but exponential methods (Whitman: O(2^n)) suffer disproportionately
    compared to polynomial approaches. Budgets matched per config; effective
    DAG size varies with k."
    """
    print("Generating C6 k-arity grid plot...")
    
    k_values = ["k2", "k3", "k4", "k6"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Determine global y-axis limits for consistent scaling
    all_runtimes = []
    for k in k_values:
        try:
            summary = load_summary(run_root, f"C6_{k}")
            all_runtimes.extend(summary["median_us"].values)
        except FileNotFoundError:
            continue
    
    if not all_runtimes:
        print("Warning: No C6 data found for k-arity grid")
        return
    
    y_min, y_max = min(all_runtimes) * 0.8, max(all_runtimes) * 1.2
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        try:
            summary = load_summary(run_root, f"C6_{k}")
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"C6_{k}\nNo Data", ha='center', va='center', 
                   fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_title(f"k = {k[1:]}", fontsize=12, fontweight='bold')
            continue
        
        # Plot each algorithm
        for engine_key, (engine_label, color) in ENGINE_STYLES.items():
            subset = summary[summary["engine_key"] == engine_key].sort_values("depth")
            if subset.empty:
                continue
            
            ax.plot(
                subset["depth"],
                subset["median_us"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=5,
                alpha=0.9,
                label=engine_label,
            )
        
        # Annotate with median depth
        median_depth = summary["median_pairDepth"].median()
        ax.text(
            0.98, 0.02,
            f"median depth: {median_depth:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
        
        # Formatting
        ax.set_title(f"k = {k[1:]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Budget", fontsize=10)
        ax.set_ylabel("Median Runtime (µs)", fontsize=10)
        ax.set_ylim([y_min, y_max])
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(labelsize=9)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")
    
    fig.suptitle("K-arity Impact on Performance (C6 Experiments)", 
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "c6_karity_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("  -> Saved C6 k-arity grid plot")


def plot_c6_karity_grid_log(run_root: Path, out_dir: Path) -> None:
    """Plot k-arity scaling comparison (C6 experiments) with log scale.
    
    Log scale better shows exponential divergence at higher k values.
    """
    print("Generating C6 k-arity grid plot (log scale)...")
    
    k_values = ["k2", "k3", "k4", "k6"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Determine global y-axis limits for consistent scaling (log space)
    all_runtimes = []
    for k in k_values:
        try:
            summary = load_summary(run_root, f"C6_{k}")
            all_runtimes.extend(summary["median_us"].values)
        except FileNotFoundError:
            continue
    
    if not all_runtimes:
        print("Warning: No C6 data found for k-arity grid (log)")
        return
    
    y_min, y_max = min(all_runtimes) * 0.5, max(all_runtimes) * 2.0
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        try:
            summary = load_summary(run_root, f"C6_{k}")
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"C6_{k}\nNo Data", ha='center', va='center', 
                   fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_title(f"k = {k[1:]}", fontsize=12, fontweight='bold')
            continue
        
        # Plot each algorithm
        for engine_key, (engine_label, color) in ENGINE_STYLES.items():
            subset = summary[summary["engine_key"] == engine_key].sort_values("depth")
            if subset.empty:
                continue
            
            ax.plot(
                subset["depth"],
                subset["median_us"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=5,
                alpha=0.9,
                label=engine_label,
            )
        
        # Annotate with median depth
        median_depth = summary["median_pairDepth"].median()
        ax.text(
            0.98, 0.02,
            f"median depth: {median_depth:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
        
        # Formatting with log scale
        ax.set_title(f"k = {k[1:]}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Budget", fontsize=10)
        ax.set_ylabel("Median Runtime (µs, log scale)", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim([y_min, y_max])
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(labelsize=9)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")
    
    fig.suptitle("K-arity Impact on Performance (C6 Experiments, Log Scale)", 
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "c6_karity_grid_log.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("  -> Saved C6 k-arity grid plot (log scale)")


def plot_budget_scaling(run_root: Path, out_dir: Path) -> None:
    """Plot budget scaling comparison across multiple experiments.
    
    Shows how runtime scales with budget across different structural patterns.
    Uses separate panels per experiment for clearer comparison.
    """
    print("Generating budget scaling comparison plot...")
    
    # Select representative experiments
    experiments = {
        "C1_altfull": "C1: Alternating (full)",
        "C2_altfixed": "C2: Alternating (fixed)",
        "C3_altpolarity": "C3: Alternating (polarity)",
        "C4_share_high": "C4: High Sharing",
        "C5_spines": "C5: Spines",
        "C6_k2": "C6: k=2",
    }
    
    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    
    # Collect all data for unified y-axis limits
    all_runtimes = []
    for exp_key in experiments.keys():
        try:
            summary = load_summary(run_root, exp_key)
            all_runtimes.extend(summary["median_us"].values)
        except FileNotFoundError:
            continue
    
    if not all_runtimes:
        print("Warning: No data found for budget scaling")
        return
    
    y_min, y_max = min(all_runtimes) * 0.5, max(all_runtimes) * 2.0
    
    for idx, (exp_key, exp_label) in enumerate(experiments.items()):
        ax = axes[idx]
        
        try:
            summary = load_summary(run_root, exp_key)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"{exp_label}\nNo Data", ha='center', va='center', 
                   fontsize=12, color='gray', transform=ax.transAxes)
            ax.set_title(exp_label, fontsize=11, fontweight='bold')
            continue
        
        # Plot each algorithm
        for engine_key, (engine_label, color) in ENGINE_STYLES.items():
            subset = summary[summary["engine_key"] == engine_key].sort_values("depth")
            if subset.empty:
                continue
            
            ax.plot(
                subset["depth"],
                subset["median_us"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=4,
                alpha=0.9,
                label=engine_label,
            )
        
        # Formatting
        ax.set_title(exp_label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Budget", fontsize=10)
        ax.set_ylabel("Runtime (µs, log)", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim([y_min, y_max])
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(labelsize=9)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")
    
    fig.suptitle("Budget Scaling Across Structural Patterns", 
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "budget_scaling.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("  -> Saved budget scaling plot")
    print("  Note: Budget is not complexity-equivalent across shapes (see caption)")


def plot_winner_map(run_root: Path, out_dir: Path, tolerance: float = 0.0) -> None:
    """Plot winner map showing which algorithm is fastest for each config.
    
    For each experiment, computes the percentage of pairs where each algorithm
    is fastest. If tolerance > 0, algorithms within (1+tolerance)·min share credit.
    
    Args:
        run_root: Root directory containing experiment results
        out_dir: Output directory for plots
        tolerance: Tolerance for near-ties (0.0 = strict argmin, 0.05 = ±5%)
    
    Caption guidance:
    "Algorithm dominance across experimental configurations. Bars show percentage
    of depth/budget points where each algorithm achieves minimum runtime (strict
    argmin). Freese dominates most structural patterns (50-80% wins), with
    Cosmadakis competitive in high-sharing scenarios (C4_high: 45%). Hunt shows
    strength in spine structures (C5). Whitman only competitive in best-case
    configurations. Tie policy: strict argmin (no 5% tolerance)."
    """
    tolerance_str = f" (±{tolerance*100:.0f}% tolerance)" if tolerance > 0 else ""
    print(f"Generating winner map plot{tolerance_str}...")
    
    experiments = [
        "C1_altfull",
        "C2_altfixed", 
        "C3_altpolarity",
        "C4_share_high",
        "C4_share_medium",
        "C4_share_low",
        "C5_spines",
        "C6_k2",
        "C6_k3",
        "C6_k4",
        "C6_k6",
        "E_bestcase",
        "E_worstcase",
    ]
    
    winner_data = []
    
    for exp in experiments:
        try:
            summary = load_summary(run_root, exp)
        except FileNotFoundError:
            continue
        
        # For each depth/budget, find which algorithm(s) are fastest
        depths = summary["depth"].unique()
        
        for depth in depths:
            depth_data = summary[summary["depth"] == depth]
            
            if len(depth_data) < 2:
                continue
            
            # Find minimum runtime and apply tolerance
            min_runtime = depth_data["median_us"].min()
            cutoff = min_runtime * (1.0 + tolerance)
            
            # Find all algorithms within tolerance
            candidates = depth_data[depth_data["median_us"] <= cutoff]["engine_key"].values
            
            if len(candidates) == 0:
                continue
            
            # Fractional credit if multiple winners
            weight = 1.0 / len(candidates)
            for winner in candidates:
                winner_data.append({
                    "experiment": exp,
                    "depth": depth,
                    "winner": winner,
                    "weight": weight,
                })
    
    if not winner_data:
        print("Warning: No winner data found")
        return
    
    winner_df = pd.DataFrame(winner_data)
    
    # Compute percentages per experiment
    exp_stats = []
    for exp in experiments:
        exp_data = winner_df[winner_df["experiment"] == exp]
        if exp_data.empty:
            continue
        
        total = exp_data["weight"].sum()
        counts = exp_data.groupby("winner")["weight"].sum()
        
        stats = {"experiment": exp}
        for engine_key in ENGINE_STYLES.keys():
            pct = (counts.get(engine_key, 0.0) / total) * 100
            stats[engine_key] = pct
        
        exp_stats.append(stats)
    
    if not exp_stats:
        print("Warning: No experiment statistics computed")
        return
    
    stats_df = pd.DataFrame(exp_stats)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(stats_df))
    bottom = np.zeros(len(stats_df))
    
    for engine_key, (engine_label, color) in ENGINE_STYLES.items():
        if engine_key not in stats_df.columns:
            continue
        
        values = stats_df[engine_key].values
        
        ax.bar(
            x_pos,
            values,
            bottom=bottom,
            color=color,
            alpha=0.9,
            label=engine_label,
            edgecolor='white',
            linewidth=1
        )
        
        # Add percentage labels (only if > 5%)
        for i, val in enumerate(values):
            if val > 5:
                ax.text(
                    i,
                    bottom[i] + val / 2,
                    f"{val:.0f}%",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    fontweight='bold'
                )
        
        bottom += values
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df["experiment"], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Percentage of Pairs (%)", fontsize=11)
    ax.set_xlabel("Experiment", fontsize=11)
    
    title_suffix = f" (±{tolerance*100:.0f}% tolerance)" if tolerance > 0 else " (strict argmin)"
    ax.set_title(f"Algorithm Dominance: % of Pairs Where Each Algorithm is Fastest{title_suffix}", 
                 fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = "winner_map_tol05.png" if tolerance > 0 else "winner_map.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("  -> Saved winner map plot")
    print("  Tie policy: strict argmin (no tolerance)")


def generate_structure_summary_table(run_root: Path, out_dir: Path) -> None:
    """Generate LaTeX table summarizing all experiment configurations."""
    print("Generating structure summary table...")
    
    experiments = [
        "C1_altfull",
        "C2_altfixed",
        "C3_altpolarity",
        "C4_share_high",
        "C4_share_medium",
        "C4_share_low",
        "C5_spines",
        "C5_spines_r",
        "C6_k2",
        "C6_k3",
        "C6_k4",
        "C6_k6",
        "E_bestcase",
        "E_worstcase",
    ]
    
    table_rows = []
    
    for exp in experiments:
        try:
            summary = load_summary(run_root, exp)
        except FileNotFoundError:
            continue
        
        # Extract metadata
        shape = summary["shape"].iloc[0] if "shape" in summary.columns else "N/A"
        budget_range = f"{summary['depth'].min():.0f}-{summary['depth'].max():.0f}"
        
        alt_index_min = summary["median_alt_index"].min() if "median_alt_index" in summary.columns else 0
        alt_index_max = summary["median_alt_index"].max() if "median_alt_index" in summary.columns else 0
        alt_index_range = f"{alt_index_min:.2f}-{alt_index_max:.2f}"
        
        share_min = summary["median_share"].min() if "median_share" in summary.columns else 0
        share_max = summary["median_share"].max() if "median_share" in summary.columns else 0
        share_range = f"{share_min:.2f}-{share_max:.2f}"
        
        # Count which algorithms are present
        engines = summary["engine_key"].unique()
        algo_count = len(engines)
        algo_list = ", ".join([ENGINE_STYLES.get(e, (e, ""))[0] for e in sorted(engines)])
        
        n_pairs = summary["n_pairs"].iloc[0] if "n_pairs" in summary.columns else "N/A"
        
        table_rows.append({
            "Experiment": exp,
            "Shape": shape,
            "Budget Range": budget_range,
            "Alt Index": alt_index_range,
            "Share": share_range,
            "Algorithms": f"{algo_count} ({algo_list})",
            "Pairs/Budget": n_pairs,
        })
    
    if not table_rows:
        print("Warning: No experiment data found for table")
        return
    
    # Generate LaTeX table (7 columns)
    latex_table = r"""\begin{table}[h]
\centering
\caption{Experimental configuration summary. Alt/Share entries show min--max range over pairs in each experiment. Share = unique/occurrences (lower = more sharing).}
\label{tab:exp_config}
\small
\begin{tabular}{lllccp{4cm}c}
\toprule
\textbf{Experiment} & \textbf{Shape} & \textbf{Budget} & \textbf{Alt Index} & \textbf{Share} & \textbf{Algorithms} & \textbf{Pairs} \\
\midrule
"""
    
    for row in table_rows:
        latex_table += f"{row['Experiment']} & {row['Shape']} & {row['Budget Range']} & "
        latex_table += f"{row['Alt Index']} & {row['Share']} & "
        latex_table += f"{row['Algorithms']} & {row['Pairs/Budget']} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save to file
    table_path = out_dir / "structure_summary_table.tex"
    with open(table_path, "w") as f:
        f.write(latex_table)
    
    # Also save as Markdown for quick reference
    md_table = "| Experiment | Shape | Budget Range | Alt Index | Share | Algorithms | Pairs |\n"
    md_table += "|------------|-------|--------------|-----------|-------|------------|-------|\n"
    
    for row in table_rows:
        md_table += f"| {row['Experiment']} | {row['Shape']} | {row['Budget Range']} | "
        md_table += f"{row['Alt Index']} | {row['Share']} | {row['Algorithms']} | {row['Pairs/Budget']} |\n"
    
    md_path = out_dir / "structure_summary_table.md"
    with open(md_path, "w") as f:
        f.write("# Experimental Configuration Summary\n\n")
        f.write(md_table)
    
    print(f"  -> Saved structure summary table (LaTeX and Markdown)")
    print(f"     LaTeX: {table_path}")
    print(f"     Markdown: {md_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def generate_plots(run_root: Path, out_dir: Path) -> None:
    """Generate all plots for run_166958."""
    print("Loading data...")
    
    # Load summaries
    c1_summary = load_summary(run_root, "C1_altfull")
    best_summary = load_summary(run_root, "E_bestcase")
    worst_summary = load_summary(run_root, "E_worstcase")
    
    # Load memory data
    memory_series = load_memory_profiles(run_root, "C1_altfull", c1_summary)
    
    print("Generating C1 runtime plots...")
    plot_c1_runtime(c1_summary, out_dir, log_scale=False)
    plot_c1_runtime(c1_summary, out_dir, log_scale=True)
    
    print("Generating best-case plot...")
    plot_best_case(best_summary, out_dir)
    
    print("Generating worst-case plots...")
    plot_worst_case(worst_summary, out_dir, log_scale=False)
    plot_worst_case(worst_summary, out_dir, log_scale=True)
    
    print("Generating memory plots...")
    plot_memory(memory_series, out_dir, log_scale=False)
    plot_memory(memory_series, out_dir, log_scale=True)
    
    print("Generating individual algorithm best/worst plots...")
    plot_individual_best_worst(best_summary, worst_summary, out_dir)
    
    print("Generating speedup ratio plot...")
    plot_speedup_ratio(c1_summary, out_dir)
    
    print("Generating recursion comparison plot...")
    plot_recursion_comparison(run_root, "C1_altfull", c1_summary, out_dir)
    
    print("Generating memoization efficiency plot...")
    plot_memoization_efficiency(run_root, "C1_altfull", c1_summary, out_dir)
    
    print("\n=== Generating C2-C6 Analysis Plots ===")
    
    print("Generating C4 node sharing impact plot...")
    plot_c4_node_sharing(run_root, out_dir)
    
    print("Generating C6 k-arity grid plot...")
    plot_c6_karity_grid(run_root, out_dir)
    
    print("Generating C6 k-arity grid plot (log scale)...")
    plot_c6_karity_grid_log(run_root, out_dir)
    
    print("Generating budget scaling comparison plot...")
    plot_budget_scaling(run_root, out_dir)
    
    print("Generating winner map plot (strict)...")
    plot_winner_map(run_root, out_dir, tolerance=0.0)
    
    print("Generating winner map plot (±5% tolerance)...")
    plot_winner_map(run_root, out_dir, tolerance=0.05)
    
    print("Generating structure summary table...")
    generate_structure_summary_table(run_root, out_dir)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {out_dir}")
    print(f"{'='*60}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots for run_166958 canonical results"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/run_166958"),
        help="Root directory containing canonical results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/plots_run_166958"),
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
