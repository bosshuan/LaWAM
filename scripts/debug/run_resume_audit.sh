#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first: conda activate vjepa2-312." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"
RUN_ID="${LAWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
AUDIT_DIR="${REPO_ROOT}/outputs/resume_audit/${RUN_ID}"
REFERENCE_DIR="${AUDIT_DIR}/uninterrupted"
RESUMED_DIR="${AUDIT_DIR}/resumed"
CONFIG="configs/debug/interndata_a1_resume_audit.yaml"

cd "${REPO_ROOT}"
"${PYTHON}" -m latent_wam.train \
  --config "${CONFIG}" \
  --output-dir "${REFERENCE_DIR}"
"${PYTHON}" -m latent_wam.train \
  --config "${CONFIG}" \
  --output-dir "${RESUMED_DIR}" \
  --stop-after 3
"${PYTHON}" -m latent_wam.train \
  --config "${CONFIG}" \
  --output-dir "${RESUMED_DIR}" \
  --resume "${RESUMED_DIR}/checkpoints/step_00000003.pt"
"${PYTHON}" -m latent_wam.resume_audit \
  --reference "${REFERENCE_DIR}/checkpoints/final.pt" \
  --candidate "${RESUMED_DIR}/checkpoints/final.pt" \
  --output "${AUDIT_DIR}/resume_audit.json"

echo "Resume audit outputs: ${AUDIT_DIR}"
