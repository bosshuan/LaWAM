#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate a Conda environment first (for example: conda activate lawam)." >&2
  exit 2
fi
PYTHON="${CONDA_PREFIX}/bin/python"

cd "${REPO_ROOT}"

CONFIG_PATH="configs/train/stage1_future.yaml"
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --config)
      if (($# < 2)); then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m latent_wam.train \
  --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
