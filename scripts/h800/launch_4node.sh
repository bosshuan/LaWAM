#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${LAWAM_REPO_ROOT:-/opt/huawei/dataset/d_env_wulan/LaWAM}"
CONDA_ROOT="${LAWAM_CONDA_ROOT:-/opt/huawei/dataset/d_env_wulan/miniconda3}"
CONDA_ENV="${LAWAM_CONDA_ENV:-${CONDA_ROOT}/envs/vjepa2-312}"
CONFIG_FILE="${LAWAM_CONFIG:-${REPO_ROOT}/configs/h800/mixture_stage1_pilot.yaml}"
MODE="${LAWAM_MODE:-preflight}"
LAWAM_CHECKPOINT="${LAWAM_CHECKPOINT:-/opt/huawei/dataset/d_env_wulan/vjepa2/checkpoints/vjepa2_1_vitG_384.pt}"
LAWAM_TEXT_MODEL="${LAWAM_TEXT_MODEL:-/opt/huawei/dataset/d_env_wulan/text/t5-large}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Set LAWAM_REPO_ROOT to the shared LaWAM repository path." >&2
  exit 2
fi
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing LaWAM config: ${CONFIG_FILE}" >&2
  exit 2
fi
: "${VC_WORKER_HOSTS:?VC_WORKER_HOSTS is required}"
: "${MA_NUM_HOSTS:?MA_NUM_HOSTS is required}"
: "${VC_TASK_INDEX:?VC_TASK_INDEX is required}"
: "${MA_NUM_GPUS:?MA_NUM_GPUS is required}"

export PATH="${CONDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/modelarts/authoring/notebook-conda/bin:/modelarts/authoring/notebook-conda/envs/jp4/bin:/opt/huawei/modelarts-dev/ma-cli/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/huawei/dataset/d_env_wulan/omnieva_3d/internvl_chat/ldd/"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON="${CONDA_ENV}/bin/python"

export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-^=mlx5_bond_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_TC="${NCCL_IB_TC:-128}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT="${LAWAM_MASTER_PORT:-29500}"
NNODES="${MA_NUM_HOSTS}"
NODE_RANK="${VC_TASK_INDEX}"
NGPUS_PER_NODE="${MA_NUM_GPUS}"
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))

if [[ "${NNODES}" -ne 4 || "${NGPUS_PER_NODE}" -ne 8 ]]; then
  echo "Expected 4 nodes x 8 GPUs, got ${NNODES} x ${NGPUS_PER_NODE}." >&2
  exit 2
fi

cd "${REPO_ROOT}"
INSTALLED_REPO="$(${PYTHON} -c 'from pathlib import Path; import latent_wam; print(Path(latent_wam.__file__).resolve().parents[2])')"
if [[ "${INSTALLED_REPO}" != "${REPO_ROOT}" ]]; then
  echo "latent_wam resolves to ${INSTALLED_REPO}, expected ${REPO_ROOT}. Run pip install -e ${REPO_ROOT}." >&2
  exit 2
fi
echo "LaWAM mode=${MODE} master=${MASTER_ADDR}:${MASTER_PORT} node_rank=${NODE_RANK} world_gpus=${TOTAL_GPUS}"

case "${MODE}" in
  preflight)
    RUN_ID="${LAWAM_RUN_ID:-latest}"
    REPORT="${REPO_ROOT}/outputs/preflight/h800_multisource/${RUN_ID}/node_${NODE_RANK}.json"
    CHECKPOINT_REPORT="${REPO_ROOT}/outputs/preflight/h800_multisource/${RUN_ID}/checkpoint_node_${NODE_RANK}.json"
    echo "H800 checkpoint audit: ${CHECKPOINT_REPORT}"
    "${PYTHON}" -m latent_wam.checkpoint_audit \
      --config "${CONFIG_FILE}" \
      --checkpoint "${LAWAM_CHECKPOINT}" \
      --output "${CHECKPOINT_REPORT}"
    echo "H800 preflight report: ${REPORT}"
    "${PYTHON}" -m latent_wam.preflight \
      --config "${CONFIG_FILE}" \
      --checkpoint "${LAWAM_CHECKPOINT}" \
      --text-model "${LAWAM_TEXT_MODEL}" \
      --expected-gpus 8 \
      --expected-device-substring H800 \
      --verify-text-model-load \
      --skip-checksum \
      --output "${REPORT}"
    ;;
  pilot)
    "${PYTHON}" -m torch.distributed.run \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NGPUS_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      -m latent_wam.train \
      --config "${CONFIG_FILE}" \
      --checkpoint "${LAWAM_CHECKPOINT}" \
      --text-model "${LAWAM_TEXT_MODEL}"
    ;;
  *)
    echo "LAWAM_MODE must be preflight or pilot, got ${MODE}." >&2
    exit 2
    ;;
esac
