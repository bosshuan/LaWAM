#!/usr/bin/env bash
set -euo pipefail

# This script runs on a CPU storage-access node. The training paths in the YAML
# remain /opt/huawei paths and are mapped only for this read-only audit.
REPO_ROOT="${LAWAM_STORAGE_REPO_ROOT:-/home/ma-user/work/dataset/d_env_wulan/LaWAM}"
CONDA_ROOT="${LAWAM_STORAGE_CONDA_ROOT:-/home/ma-user/work/dataset/d_env_wulan/miniconda3}"
CONDA_ENV="${LAWAM_STORAGE_CONDA_ENV:-${CONDA_ROOT}/envs/vjepa2-312}"
PYTHON="${LAWAM_STORAGE_PYTHON:-${CONDA_ENV}/bin/python3.12}"
CONFIG_FILE="${LAWAM_CONFIG:-${REPO_ROOT}/configs/h800/mixture_stage1_pilot.yaml}"
RUN_ID="${LAWAM_RUN_ID:-latest}"
REPORT="${LAWAM_MANIFEST_REPORT:-${REPO_ROOT}/outputs/preflight/storage_manifest/${RUN_ID}.json}"
DETAIL_REPORT="${LAWAM_MANIFEST_DETAIL_REPORT:-${REPO_ROOT}/outputs/preflight/storage_manifest/${RUN_ID}.full.json}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Missing storage-view LaWAM repository: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing storage-view Python interpreter: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing LaWAM config: ${CONFIG_FILE}" >&2
  exit 2
fi

# The storage view can coexist with an editable install that points at the
# training-time /opt/huawei mount. Conda's launcher also has an absolute
# /opt/huawei shebang, so bypass activation and invoke the environment's ELF
# interpreter directly.
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CONDA_PREFIX="${CONDA_ENV}"
export CONDA_DEFAULT_ENV="vjepa2-312"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""

cd "${REPO_ROOT}"
echo "CPU storage manifest detailed report: ${DETAIL_REPORT}"
echo "CPU storage manifest compact report: ${REPORT}"
set +e
"${PYTHON}" -m latent_wam.manifest_audit \
  --config "${CONFIG_FILE}" \
  --path-map-from /opt/huawei \
  --path-map-to /home/ma-user/work \
  --output "${DETAIL_REPORT}"
AUDIT_STATUS=$?
set -e

"${PYTHON}" -m latent_wam.manifest_compact \
  --input "${DETAIL_REPORT}" \
  --output "${REPORT}" \
  --detail-source oxe \
  --detail-source robomind

exit "${AUDIT_STATUS}"
