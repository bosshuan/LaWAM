#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${LAWAM_STORAGE_REPO_ROOT:-/home/ma-user/work/dataset/d_env_wulan/LaWAM}"
CONDA_ROOT="${LAWAM_STORAGE_CONDA_ROOT:-/home/ma-user/work/dataset/d_env_wulan/miniconda3}"
CONDA_ENV="${LAWAM_STORAGE_CONDA_ENV:-${CONDA_ROOT}/envs/vjepa2-312}"
PYTHON="${LAWAM_STORAGE_PYTHON:-${CONDA_ENV}/bin/python3.12}"
RUN_ID="${LAWAM_RUN_ID:-latest}"
INPUT="${LAWAM_MANIFEST_DETAIL_REPORT:-${REPO_ROOT}/outputs/preflight/storage_manifest/${RUN_ID}.json}"
OUTPUT="${LAWAM_MANIFEST_COMPACT_REPORT:-${REPO_ROOT}/outputs/preflight/storage_manifest/${RUN_ID}-compact.json}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing storage-view Python interpreter: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "Missing detailed storage manifest: ${INPUT}" >&2
  exit 2
fi
if [[ "${INPUT}" == "${OUTPUT}" ]]; then
  echo "Compact output must differ from detailed input: ${INPUT}" >&2
  exit 2
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CONDA_PREFIX="${CONDA_ENV}"
export CONDA_DEFAULT_ENV="vjepa2-312"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
"${PYTHON}" -m latent_wam.manifest_compact \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --detail-source oxe \
  --detail-source robomind
