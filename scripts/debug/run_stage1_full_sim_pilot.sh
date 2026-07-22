#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first: conda activate vjepa2-312." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"
RUN_ID="${LAWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${REPO_ROOT}/outputs/interndata_a1_stage1_full_sim_pilot/${RUN_ID}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "${REPO_ROOT}"
echo "Stage 1 full-sim pilot output: ${OUTPUT_DIR}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m latent_wam.train \
  --config configs/debug/interndata_a1_stage1_full_sim_pilot.yaml \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
