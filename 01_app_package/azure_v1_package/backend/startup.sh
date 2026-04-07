#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/site/wwwroot/app}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

export PYTHONPATH="${APP_ROOT}/backend:${APP_ROOT}:${PYTHONPATH:-}"
export IWERP_BASE_DIR="${IWERP_BASE_DIR:-${APP_ROOT}}"

cd "${APP_ROOT}"
exec "${PYTHON_BIN}" -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
