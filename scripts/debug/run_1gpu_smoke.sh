#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first (for example: conda activate lawam)." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
"${PYTHON}" -m latent_wam.train \
  --config configs/debug/interndata_a1_8gpu_smoke.yaml \
  --max-steps 2 "$@"
