#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/llama.cpp/llama-server}"
MODEL_GGUF_PATH="${MODEL_GGUF_PATH:-./gguf/master_sovereign_v51_gold_q4_k_m.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-8}"

exec "${LLAMA_SERVER_BIN}" \
  --model "${MODEL_GGUF_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --ctx-size "${CTX_SIZE}" \
  --threads "${THREADS}" \
  --alias "IWFUSION-SLM-V1"

