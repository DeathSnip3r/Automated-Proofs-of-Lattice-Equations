#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
  RUNNER_PY="${SLURM_SUBMIT_DIR}/scripts/run.py"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  RUNNER_PY="${SCRIPT_DIR}/run.py"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'ERROR: python3 not found on PATH.\n' >&2
  exit 1
fi

OUT_ROOT_ARG=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUT_ROOT_ARG=("--out-root" "results/run_${SLURM_JOB_ID}")
fi

exec "$PYTHON_BIN" "$RUNNER_PY" "${OUT_ROOT_ARG[@]}" "$@"
