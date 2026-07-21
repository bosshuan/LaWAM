#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first: conda activate vjepa2-312." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m latent_wam.train \
  --config configs/debug/interndata_a1_8gpu_smoke.yaml "$@"
