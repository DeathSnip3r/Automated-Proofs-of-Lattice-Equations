# Automated Proofs of Lattice Equations

## Project Overview
This repository accompanies an empirical study of automated decision procedures for lattice equations. It implements and benchmarks four classical algorithms—Whitman, Freese, Cosmadakis, and Hunt—against synthetically generated term pairs under controlled structural parameters.

## Repository Layout
- `Makefile` – top-level build targets for compiling the command-line tools and running experiments.
- `src/` – C++ sources for the decision procedures, shared infrastructure (interners, generators), and runners.
- `include/` – public headers shared across the engines and experiment tooling.
- `bin/` – compiled executables (e.g., `check.exe`, `lattice_gen.exe`) produced by the build.
- `build/` – intermediate object files generated during compilation.
- `analysis/` – notebooks, scripts, and generated artefacts used to post-process experiment logs and create figures.
- `results/` – raw CSV outputs and run directories produced by the experimental harness.
- `scripts/` – helper scripts (PowerShell and Python) for batch generation, plotting, and automation.
- `trash/` – archival experiment runs and prototype drivers retained for reference (not part of the main pipeline).

## Building and Running
1. `make` – builds the core executables into `bin/` and intermediate objects into `build/`.
2. `make clean` – removes compiled artefacts.
3. `scripts/run.sh` - run entire pipeline with preconfigurable parameters in `run.py`
Experiment scripts under `scripts/` orchestrate large benchmarking batches and write results into `results/`. Generated plots and post-processing summaries live in the `analysis/` subdirectories.
