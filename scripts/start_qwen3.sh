#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Model
# Hugging Face repo ID
# 已缓存时会直接使用 ~/.cache/huggingface 中的模型
# ============================================================
MODEL="Inferact/Qwen3.8-27B-NVFP4"
SERVED_NAME="Qwen3.8-27B"

# ============================================================
# Server
# ============================================================
HOST="0.0.0.0"
PORT=8000

# RTX PRO 5000 72GB
# MAX_MODEL_LEN=131072
# GPU_MEMORY_UTILIZATION=0.85
# MAX_NUM_SEQS=16

MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.7
MAX_NUM_SEQS=2

# ============================================================
# CUDA 12.9
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
# ============================================================
exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype fp8 \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    "$@"