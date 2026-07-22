#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first: conda activate vjepa2-312." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"
RUN_ID="${LAWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
REPORT="${REPO_ROOT}/outputs/preflight/t5_a100_${RUN_ID}.json"

cd "${REPO_ROOT}"
echo "T5 A100 preflight report: ${REPORT}"
"${PYTHON}" -m latent_wam.preflight \
  --config configs/debug/interndata_a1_8gpu_t5_smoke.yaml \
  --verify-text-model-load \
  --output "${REPORT}" \
  "$@"
