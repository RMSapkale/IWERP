#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./smoke_test_api.sh <base_url> <username> <password> [tenant_name]
# Example:
#   ./smoke_test_api.sh https://api.example.com smoke_user 'StrongPass#1' 'Smoke Tenant'

BASE_URL="${1:-}"
USERNAME="${2:-}"
PASSWORD="${3:-}"
TENANT_NAME="${4:-V1 Smoke Tenant}"

if [[ -z "${BASE_URL}" || -z "${USERNAME}" || -z "${PASSWORD}" ]]; then
  echo "Usage: $0 <base_url> <username> <password> [tenant_name]"
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for this smoke script"
  exit 2
fi

echo "[1/4] Health check"
curl -fsS "${BASE_URL}/health" | jq .

echo "[2/4] Register/login"
REG_PAYLOAD="$(jq -n --arg u "${USERNAME}" --arg p "${PASSWORD}" --arg t "${TENANT_NAME}" '{username:$u,password:$p,tenant_name:$t}')"
REG_CODE="$(curl -sS -o /tmp/iwerp_register_resp.json -w "%{http_code}" -X POST "${BASE_URL}/v1/auth/register" -H 'Content-Type: application/json' -d "${REG_PAYLOAD}")"
if [[ "${REG_CODE}" == "200" ]]; then
  AUTH_RESP="$(cat /tmp/iwerp_register_resp.json)"
elif [[ "${REG_CODE}" == "409" ]]; then
  LOGIN_PAYLOAD="$(jq -n --arg u "${USERNAME}" --arg p "${PASSWORD}" '{username:$u,password:$p}')"
  LOGIN_CODE="$(curl -sS -o /tmp/iwerp_login_resp.json -w "%{http_code}" -X POST "${BASE_URL}/v1/auth/login" -H 'Content-Type: application/json' -d "${LOGIN_PAYLOAD}")"
  if [[ "${LOGIN_CODE}" != "200" ]]; then
    RETRY_SUFFIX="$(date +%s)"
    RETRY_USERNAME="${USERNAME}_${RETRY_SUFFIX}"
    RETRY_TENANT="${TENANT_NAME} ${RETRY_SUFFIX}"
    echo "existing username not reusable; retrying with ${RETRY_USERNAME}"
    RETRY_PAYLOAD="$(jq -n --arg u "${RETRY_USERNAME}" --arg p "${PASSWORD}" --arg t "${RETRY_TENANT}" '{username:$u,password:$p,tenant_name:$t}')"
    RETRY_CODE="$(curl -sS -o /tmp/iwerp_register_resp_retry.json -w "%{http_code}" -X POST "${BASE_URL}/v1/auth/register" -H 'Content-Type: application/json' -d "${RETRY_PAYLOAD}")"
    if [[ "${RETRY_CODE}" != "200" ]]; then
      echo "retry register failed with status=${RETRY_CODE}"
      cat /tmp/iwerp_register_resp_retry.json
      exit 1
    fi
    AUTH_RESP="$(cat /tmp/iwerp_register_resp_retry.json)"
  else
    AUTH_RESP="$(cat /tmp/iwerp_login_resp.json)"
  fi
else
  echo "register failed with status=${REG_CODE}"
  cat /tmp/iwerp_register_resp.json
  exit 1
fi

TOKEN="$(echo "${AUTH_RESP}" | jq -r '.access_token // empty')"
if [[ -z "${TOKEN}" ]]; then
  echo "register/login failed: missing access_token"
  echo "${AUTH_RESP}" | jq .
  exit 1
fi

call_chat() {
  local prompt="$1"
  local payload
  payload="$(jq -n --arg p "${prompt}" '{messages:[{role:"user",content:$p}],debug:false}')"
  curl -fsS -X POST "${BASE_URL}/v1/sovereign/chat/completions" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H 'Content-Type: application/json' \
    -d "${payload}"
}

echo "[3/4] Sovereign inference smoke prompts"
echo "summary: EPM"
call_chat "what is EPM?" | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "summary: GL"
call_chat "what is GL?" | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "summary: RMCS"
call_chat "what is RMCS?" | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "guarded summary: payroll gratuity"
call_chat "what is payroll gratuity?" | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "procedure: custom ESS job"
call_chat "how to create custom ESS job?" | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "guarded SQL: AP invoice distributions full-shape"
call_chat "Create an Oracle Fusion SQL query to extract AP invoice distribution details. Include Invoice Number, Supplier Name, Distribution Line Number, Distribution Amount, Natural Account Segment, Cost Center, Liability Account. Show only validated and accounted invoices." \
  | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "fast formula: sick leave accrual"
call_chat "Create a Fast Formula for sick leave accrual with DEFAULT handling, INPUTS, and RETURN logic." \
  | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'
echo "guarded fast formula: payroll gratuity"
call_chat "Create a Fast Formula for payroll gratuity eligibility and payout logic with explicit RETURN." \
  | jq '{task_type,selected_module,verifier_status,refusal,citations:(.citations|length)}'

echo "[4/4] Sovereign smoke complete."
