# DevOps Deployment Checklist

Use this checklist from the handoff root.

## 1. Package integrity

- Confirm `01_app_package/azure_v1_package/app/backend/main.py` exists
- Confirm `01_app_package/azure_v1_package/app/frontend/index.html` exists
- Confirm `02_model_runtime/gguf/master_sovereign_v51_gold_q4_k_m.gguf` exists
- Confirm `02_model_runtime/mlx_adapter/adapters.safetensors` exists
- Confirm `runtime_content/iwerp-prod/` exists
- Confirm `docker-compose.yml` exists
- Confirm `docker/nginx/nginx.conf` exists
- Confirm `03_validation_evidence/` exists

## 2. Host prerequisites

- Docker Engine installed
- Docker Compose v2 available
- Enough disk for images, model, and postgres volume
- Enough RAM/CPU for `llama.cpp` model serving
- Public DNS for `iwerp.com` ready

## 3. Secrets and env

- Copy `.env.docker.example` to `.env`
- Set `SECRET_KEY`
- Set `POSTGRES_PASSWORD`
- Confirm `ALLOWED_ORIGINS=https://iwerp.com`
- Confirm `VITE_API_BASE_URL=https://iwerp.com/api`
- Confirm `IWFUSION_ENABLE_LEGACY_OPENAI_ROUTES=false`

## 4. TLS

- Production: place real certs in `docker/nginx/certs/`
- Local smoke only: generate self-signed certs with `./docker/nginx/generate-self-signed.sh`

## 5. Build and bootstrap

- Run `docker compose build`
- Run `docker compose --profile init run --rm db-bootstrap`
- Confirm bootstrap completed in safe mode
- Do not use reset mode on existing environments

## 6. Start services

- Run `docker compose up -d`
- Confirm services are up:
  - `proxy`
  - `frontend`
  - `backend`
  - `model`
  - `postgres`
  - `redis`

## 7. Health and route checks

- `curl -k -i https://iwerp.com/api/health` returns `200`
- `curl -k -i https://iwerp.com/api/v1/sovereign/chat/completions` returns `401` without auth
- Confirm UI loads on `https://iwerp.com`
- Confirm public docs/UI do not expose `/v1/openai/*`

## 8. Smoke prompts

Run the packaged prompt checks from:

- `01_app_package/azure_v1_package/checks/smoke_test_prompts.md`
- `01_app_package/azure_v1_package/checks/smoke_test_api.sh`

Minimum prompts:

- `what is EPM?`
- `what is RMCS?`
- `what is GL?`
- `what is payroll gratuity?`
- `how to create custom ESS job?`
- representative AP / AR / GL / Procurement SQL prompts

## 9. Logs

- Backend logs: `docker compose logs -f backend`
- SQL decision logs: `docker compose logs -f backend | grep sql_decision_event`
- Proxy logs: `docker compose logs -f proxy`

## 10. Evidence review

- Review `03_validation_evidence/v1_ui_local_validation_report.json`
- Review `03_validation_evidence/sql_validation_engine_direct/cross_module_sql_showcase_summary.json`
- Review `03_validation_evidence/sql_validation_live_api/cross_module_sql_api_path_summary.json`

## 11. External items still required

- Real TLS certs for internet deployment
- DNS record for `iwerp.com`
- Host/network firewall and LB policy
- Monitoring, backups, and secret storage according to your environment
