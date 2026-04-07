# IWFUSION-V1 Final DevOps Handoff

This folder is the final Docker-ready and DevOps-ready handoff root for `iwfusion-v1`.

Use this folder as the deployment root.

## What is in this handoff

- application source: `01_app_package/azure_v1_package/app/backend` and `01_app_package/azure_v1_package/app/frontend`
- model runtime: `02_model_runtime/`
- packaged grounding/runtime data: `runtime_content/`
- Docker deployment stack: `docker-compose.yml`, `.env.docker.example`, `docker/`
- validation evidence: `03_validation_evidence/`
- top-level operator docs:
  - `DOCKER_DEPLOYMENT_README.md`
  - `DEVOPS_DEPLOYMENT_CHECKLIST.md`
  - `FINAL_MISSING_ITEMS_CHECK.md`
  - `HANDOFF_FOLDER_TREE.txt`

## Exact bring-up order

1. Start from this root:

```bash
cd drive_handoff_iwfusion_v1_20260406
```

2. Copy the Docker env template:

```bash
cp .env.docker.example .env
```

3. Edit `.env` and set at minimum:

- `SECRET_KEY`
- `POSTGRES_PASSWORD`
- `ALLOWED_ORIGINS=https://iwerp.com`
- `VITE_API_BASE_URL=https://iwerp.com/api`
- any host-specific port overrides if needed

4. Provide TLS certs for `iwerp.com`:

- production: place real certs at `docker/nginx/certs/fullchain.pem` and `docker/nginx/certs/privkey.pem`
- local smoke only: run `./docker/nginx/generate-self-signed.sh`

5. Confirm required payloads exist:

- app source under `01_app_package/azure_v1_package/app/`
- model file under `02_model_runtime/gguf/master_sovereign_v51_gold_q4_k_m.gguf`
- runtime data under `runtime_content/iwerp-prod/`

6. Build containers:

```bash
docker compose build
```

7. Run one-time safe DB bootstrap:

```bash
docker compose --profile init run --rm db-bootstrap
```

Default mode is safe migrate. It creates missing tables and applies additive schema updates without dropping existing data.

8. Start the stack:

```bash
docker compose up -d
```

9. Smoke-test public routes:

```bash
curl -k -i https://iwerp.com/api/health
curl -k -i https://iwerp.com/api/v1/sovereign/chat/completions
```

Expected:

- `/api/health` -> `200`
- `/api/v1/sovereign/chat/completions` -> `401` without auth

10. Run prompt smoke tests from:

- `01_app_package/azure_v1_package/checks/smoke_test_prompts.md`
- `01_app_package/azure_v1_package/checks/smoke_test_api.sh`

## Public contract

Public routes intended for UI and clients:

- `/v1/sovereign/chat/completions`
- `/v1/sovereign/responses`

Legacy `/v1/openai/*` compatibility routes are not part of the public contract and are disabled by default in production via `IWFUSION_ENABLE_LEGACY_OPENAI_ROUTES=false`.

## Recommended files for DevOps first

Read in this order:

1. `README_FIRST.md`
2. `DEVOPS_DEPLOYMENT_CHECKLIST.md`
3. `DOCKER_DEPLOYMENT_README.md`
4. `FINAL_MISSING_ITEMS_CHECK.md`

## Validation evidence included

- UI/API local validation:
  - `03_validation_evidence/v1_ui_local_validation_report.json`
- SQL engine-direct validation:
  - `03_validation_evidence/sql_validation_engine_direct/`
- SQL live API-path validation:
  - `03_validation_evidence/sql_validation_live_api/`

## Final package status

- handoff root is complete for Docker deployment
- no package-critical runtime artifact is missing
- external production prerequisites still apply:
  - Docker host
  - DNS for `iwerp.com`
  - real TLS certificates
  - firewall / LB rules
