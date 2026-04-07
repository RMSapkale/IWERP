#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="/home/site/wwwroot/app/frontend"
PORT="${PORT:-3001}"
cd "${APP_ROOT}"

if [ ! -d "node_modules" ]; then
  npm ci
fi

npm run build
npm run preview -- --host 0.0.0.0 --port "${PORT}"
