#!/usr/bin/env bash
set -euo pipefail

# Example only: fill values before use.
RG="<resource-group>"
LOC="<location>"
PLAN="<appservice-plan>"
BACKEND_APP="<backend-app-name>"
FRONTEND_APP="<frontend-app-name>"
RUNTIME_PY="PYTHON|3.11"
RUNTIME_NODE="NODE|20-lts"
FRONTEND_URL="https://iwerp.com"
BACKEND_URL="https://iwerp.com/api"

az group create -n "${RG}" -l "${LOC}"
az appservice plan create -g "${RG}" -n "${PLAN}" --is-linux --sku B2

az webapp create -g "${RG}" -p "${PLAN}" -n "${BACKEND_APP}" --runtime "${RUNTIME_PY}"
az webapp create -g "${RG}" -p "${PLAN}" -n "${FRONTEND_APP}" --runtime "${RUNTIME_NODE}"

az webapp config appsettings set -g "${RG}" -n "${BACKEND_APP}" --settings \
  IWERP_BASE_DIR="/home/site/wwwroot/app" \
  ALLOWED_ORIGINS="${FRONTEND_URL}" \
  IWFUSION_INFERENCE_BACKEND="llama_cpp"

az webapp config appsettings set -g "${RG}" -n "${FRONTEND_APP}" --settings \
  VITE_API_BASE_URL="${BACKEND_URL}" \
  VITE_ENABLE_ADMIN_DEBUG="0" \
  VITE_ENABLE_PLSQL="0"

az webapp config set -g "${RG}" -n "${BACKEND_APP}" --startup-file "azure/appservice_backend_startup.sh"
az webapp config set -g "${RG}" -n "${FRONTEND_APP}" --startup-file "azure/appservice_frontend_startup.sh"

echo "Deploy code with your preferred method:"
echo "  az webapp deploy -g ${RG} -n ${BACKEND_APP} --src-path <backend-zip-or-folder>"
echo "  az webapp deploy -g ${RG} -n ${FRONTEND_APP} --src-path <frontend-zip-or-folder>"
