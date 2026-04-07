#!/usr/bin/env sh
set -eu

MODEL_PATH="${MODEL_PATH:-/models/master_sovereign_v51_gold_q4_k_m.gguf}"
MODEL_CONTEXT_SIZE="${MODEL_CONTEXT_SIZE:-8192}"
MODEL_THREADS="${MODEL_THREADS:-8}"
MODEL_GPU_LAYERS="${MODEL_GPU_LAYERS:-0}"
MODEL_PORT_INTERNAL="${MODEL_PORT_INTERNAL:-8080}"

exec llama-server \
  --host 0.0.0.0 \
  --port "${MODEL_PORT_INTERNAL}" \
  --model "${MODEL_PATH}" \
  --ctx-size "${MODEL_CONTEXT_SIZE}" \
  --threads "${MODEL_THREADS}" \
  --n-gpu-layers "${MODEL_GPU_LAYERS}"
