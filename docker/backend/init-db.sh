#!/usr/bin/env sh
set -eu

MODE="${DB_INIT_MODE:-safe}"
RESET_CONFIRM="${DB_RESET_CONFIRM:-}"

case "${MODE}" in
  safe)
    echo "Running safe database bootstrap."
    echo "Behavior: creates missing tables, applies additive/idempotent upgrades, preserves existing data."
    exec python -m backend.core.database.migrate --mode safe
    ;;
  reset)
    echo "Running destructive reset mode for demo/dev."
    echo "Behavior: drops and recreates application tables before reseeding."
    if [ "${RESET_CONFIRM}" != "YES_RESET" ]; then
      echo "Refusing reset. Re-run with DB_INIT_MODE=reset and DB_RESET_CONFIRM=YES_RESET."
      exit 2
    fi
    exec python -m backend.core.database.migrate --mode reset --force-reset YES_RESET
    ;;
  *)
    echo "Unsupported DB_INIT_MODE='${MODE}'. Expected 'safe' or 'reset'."
    exit 2
    ;;
esac
