#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first: conda activate vjepa2-312." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"
RUN_ID="${LAWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SUITE_DIR="${REPO_ROOT}/outputs/strict_overfit/${RUN_ID}"

cd "${REPO_ROOT}"
"${PYTHON}" -m latent_wam.train \
  --config configs/debug/interndata_a1_future_overfit.yaml \
  --output-dir "${SUITE_DIR}/future"
"${PYTHON}" -m latent_wam.train \
  --config configs/debug/interndata_a1_action_overfit.yaml \
  --output-dir "${SUITE_DIR}/action"
"${PYTHON}" -m latent_wam.train \
  --config configs/debug/interndata_a1_tiny_overfit.yaml \
  --output-dir "${SUITE_DIR}/joint"

echo "Strict overfit outputs: ${SUITE_DIR}"
