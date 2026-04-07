# Azure Release Checklist (V1)

## 1) Required Environment Variables

- `DATABASE_URL`
- `SECRET_KEY`
- `LOCAL_SLM_BASE_URL`
- `ALLOWED_ORIGINS`
- `IWERP_BASE_DIR`

Default deployment values used in this package:
- frontend URL: `https://iwerp.com`
- backend URL: `https://iwerp.com/api`
- `ALLOWED_ORIGINS=https://iwerp.com`

## 2) Recommended Environment Variables

- `IWFUSION_INFERENCE_BACKEND=llama_cpp`
- `IWFUSION_MODEL_NAME=IWFUSION-SLM-V1`
- `IWFUSION_USE_RERANKER=false`
- `IWFUSION_USE_EMBEDDINGS=true`
- `LLAMA_CPP_BASE_URL` (if inference host differs)

## 3) Secrets and Access

- JWT secret is set (`SECRET_KEY`) and not committed in repo.
- DB credentials are stored in Azure App Settings / Key Vault reference.
- Inference endpoint credentials (if any) are stored as secrets.

## 4) Storage / Index Prerequisites

- `IWERP_BASE_DIR` points to mounted runtime content root.
- Required corpora/index folders exist under that root.
- File permissions allow backend read access.

## 5) Backend Startup Check

1. Deploy backend code/package.
2. Run startup command:
`azure/appservice_backend_startup.sh`
3. Verify:
- `GET /health` => 200
- `POST /v1/sovereign/chat/completions` (without auth) => 401

## 6) Frontend Startup Check

1. Set `VITE_API_BASE_URL` to `https://iwerp.com/api`.
2. Build and deploy frontend.
3. Verify app loads and login/register works.

Package source paths:
- backend app: `app/backend`
- frontend app: `app/frontend`
- MOA route: `app/backend/moa_routes.py`
- MOE runtime: `app/backend/core/moe/`

## 7) Sovereign Endpoint Verification

- UI requests only:
  - `/v1/sovereign/chat/completions`
  - `/v1/sovereign/responses`
- Public docs/UI do not advertise legacy compatibility routes.

## 8) Health and Readiness

- Unauthenticated health probe: `GET /health`
- Authenticated readiness probe: `GET /v1/health/readiness` with valid JWT/API key

## 9) Smoke Test Prompts (Post Deploy)

Run via UI or `checks/smoke_test_api.sh`:
- Summary: `what is EPM?`
- Summary: `what is GL?`
- Summary: `what is RMCS?`
- Guarded summary: `what is payroll gratuity?`
- Procedure: `how to create custom ESS job?`
- SQL: full-shape AP invoice distributions query prompt
- Fast Formula: accrual formula + troubleshooting prompt

Expected outcomes:
- `what is payroll gratuity?` may return a clean refusal if exact concept grounding is absent.
- Full-shape AP invoice distribution SQL may return a safe refusal if complete grounded joins/fields/filters are not available.

## 10) Rollback Notes

- Keep previous backend slot/image ready.
- Keep previous frontend build artifact ready.
- Rollback trigger examples:
  - sovereign route 5xx spike
  - auth failures
  - SQL/FF verifier failures above normal baseline

## 11) Known Limitations / Watchlist

- Guarded refusals remain part of the validated contract for weakly grounded gratuity and full-shape SQL prompts.
- `/v1/health/readiness` requires auth by design.
