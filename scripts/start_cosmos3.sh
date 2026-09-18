#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Model — Cosmos3-Nano unified omni checkpoint (local path)
# vLLM serves it via the native Cosmos3ForConditionalGeneration
# architecture; the reasoner path handles VLM Q&A.
# ============================================================
MODEL="/mnt/datadisk/models/Cosmos3-Nano"
SERVED_NAME="Cosmos3-Nano"

# ============================================================
# Server
# ============================================================
HOST="0.0.0.0"
PORT=8000

# RTX PRO 5000 72GB, omni weights ~34GB
# 0.80 x 71GB = ~57GB (0.92 default OOMs on this card)
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.70
MAX_NUM_SEQS=2

# ============================================================
# CUDA 12.9 (SM 12.x 必需：默认 12.8 认不出 Blackwell，
# 会导致 FlashInfer arch 探测失败。Qwen 脚本同理。)
# 只影响当前脚本启动的 vLLM，不改变系统默认 CUDA 12.8
# ============================================================
export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# 单卡
export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Conda
# ============================================================
eval "$(conda shell.bash hook)"
conda activate vllm

# ============================================================
# Environment check
# ============================================================
echo "============================================================"
echo " Model      : ${MODEL}"
echo " CUDA_HOME  : ${CUDA_HOME}"
echo " Context    : ${MAX_MODEL_LEN}"
echo " GPU Memory : ${GPU_MEMORY_UTILIZATION}"
echo "============================================================"

echo
echo "[CUDA]"
which nvcc
nvcc --version | grep "release"

echo
echo "[PyTorch / GPU]"
python - <<'PY'
import torch

print("torch       =", torch.__version__)
print("torch CUDA  =", torch.version.cuda)
print("GPU         =", torch.cuda.get_device_name(0))
print("SM          =", torch.cuda.get_device_capability(0))
PY

echo
echo "[FlashInfer]"
python -m flashinfer show-config 2>/dev/null | \
    grep -E "FlashInfer version|FLASHINFER_CUDA_VERSION|FLASHINFER_CUDA_ARCH_LIST|CUDA_VERSION|CUDA_HOME|NVCC" \
    || true

echo
echo "============================================================"
echo "Starting vLLM..."
echo "============================================================"

# ============================================================
# Start vLLM
# --mm-encoder-tp-mode/--async-scheduling/--media-io-kwargs:
# official Cosmos3 serving flags (mm-encoder flag is a no-op
# at tensor-parallel-size 1; video kwargs kept for future
# temporal inputs — current policy sends single images).
# NOTE: no --tool-call-parser/--reasoning-parser/--speculative-config:
# those are Qwen3-Coder specific and unsupported here.
# ============================================================
exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --allowed-local-media-path / \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
