#!/usr/bin/env bash
# Local iteration helper: run a command inside the pixi environment without
# going through `pixi run` (which spawns a tokio runtime this box often lacks
# the pids budget for). All repository tasks still go through `pixi run`.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo_root/.pixi/envs/default"
export PATH="$env_dir/bin:$PATH"
export MODULAR_HOME="$env_dir/share/max"
export CONDA_PREFIX="$env_dir"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec "$@"