#!/usr/bin/env bash
set -euo pipefail

# This script runs on a CPU storage-access node. The training paths in the YAML
# remain /opt/huawei paths and are mapped only for this read-only audit.
REPO_ROOT="${LAWAM_STORAGE_REPO_ROOT:-/home/ma-user/work/dataset/d_env_wulan/LaWAM}"
CONDA_ROOT="${LAWAM_STORAGE_CONDA_ROOT:-/home/ma-user/work/dataset/d_env_wulan/miniconda3}"
CONDA_ENV="${LAWAM_STORAGE_CONDA_ENV:-${CONDA_ROOT}/envs/vjepa2-312}"
CONFIG_FILE="${LAWAM_CONFIG:-${REPO_ROOT}/configs/h800/mixture_stage1_pilot.yaml}"
RUN_ID="${LAWAM_RUN_ID:-latest}"
REPORT="${LAWAM_MANIFEST_REPORT:-${REPO_ROOT}/outputs/preflight/storage_manifest/${RUN_ID}.json}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Missing storage-view LaWAM repository: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda initialization: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing LaWAM config: ${CONFIG_FILE}" >&2
  exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON="${CONDA_ENV}/bin/python"

# The storage view can coexist with an editable install that points at the
# training-time /opt/huawei mount, so select this checkout explicitly.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""

cd "${REPO_ROOT}"
echo "CPU storage manifest report: ${REPORT}"
"${PYTHON}" -m latent_wam.manifest_audit \
  --config "${CONFIG_FILE}" \
  --path-map-from /opt/huawei \
  --path-map-to /home/ma-user/work \
  --output "${REPORT}"
