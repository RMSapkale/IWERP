# IWFUSION-V1 Docker Deployment

This stack deploys the full application from the Drive handoff root:

- nginx reverse proxy / TLS edge
- frontend
- backend
- model runtime
- PostgreSQL
- Redis
- packaged runtime content required by the backend

## Included Docker Assets

- `docker-compose.yml`
- `.env.docker.example`
- `docker/nginx/nginx.conf`
- `docker/nginx/generate-self-signed.sh`
- `docker/nginx/certs/`
- `docker/backend/init-db.sh`
- `docker/model/start-model.sh`
- `01_app_package/azure_v1_package/` application package
- `02_model_runtime/` model assets
- `runtime_content/` packaged grounding/runtime assets

## URL Contract

- frontend: `https://iwerp.com`
- backend API: `https://iwerp.com/api`

The compose stack exposes:

- nginx edge proxy on host ports `80` and `443`
- backend on host port `8000`
- model on host port `8080`
- postgres on host port `5432`
- redis on host port `6379`

## First-Time Setup

1. Copy the env template:

```bash
cp .env.docker.example .env
```

2. Review and update at minimum:

- `SECRET_KEY`
- `POSTGRES_PASSWORD`
- `ALLOWED_ORIGINS`
- `VITE_API_BASE_URL`

3. For local TLS bring-up, generate a self-signed certificate:

```bash
chmod +x docker/nginx/generate-self-signed.sh
./docker/nginx/generate-self-signed.sh
```

For production, replace `docker/nginx/certs/fullchain.pem` and `docker/nginx/certs/privkey.pem` with real certificates for `iwerp.com`.

4. Build the stack:

```bash
docker compose build
```

5. Bootstrap the database once:

```bash
docker compose --profile init run --rm db-bootstrap
```

Safe mode behavior:

- creates missing tables
- applies additive and idempotent schema upgrades
- preserves existing data
- ensures the base `demo` tenant exists

Optional destructive reset for demo/dev only:

```bash
DB_INIT_MODE=reset DB_RESET_CONFIRM=YES_RESET \
docker compose --profile init run --rm db-bootstrap
```

Reset mode behavior:

- drops and recreates the application tables
- reseeds the base `demo` tenant
- should not be used on existing production data

6. Start the services:

```bash
docker compose up -d
```

## Traffic Routing

- `https://iwerp.com/` -> `frontend` container
- `https://iwerp.com/api/` -> `backend` container

The backend service still listens internally on port `8000`, and its direct host port mapping remains `8000:8000` for admin/debug use if needed.

## Runtime Paths

- app package: `01_app_package/azure_v1_package`
- model file: `02_model_runtime/gguf/master_sovereign_v51_gold_q4_k_m.gguf`
- runtime content mount inside backend: `/app/runtime`
- backend `IWERP_BASE_DIR`: `/app/runtime/iwerp-prod`

## Health Checks

After startup:

```bash
curl -k -i https://iwerp.com/api/health
curl -k -i https://iwerp.com/api/v1/sovereign/chat/completions
```

The sovereign endpoint should answer `401` without auth.

## Important Notes

- The `db-bootstrap` service is intentionally separate and now defaults to safe migration mode.
- Destructive reset remains available only with `DB_INIT_MODE=reset` and `DB_RESET_CONFIRM=YES_RESET`.
- The model service uses the packaged GGUF artifact and `llama.cpp` server image.
- The backend depends on `runtime_content/` for procedure, troubleshooting, summary, and SQL grounding assets.
- The reverse proxy is now included in the Docker stack.
- For production, real TLS certificates for `iwerp.com` are still external input and must be mounted into `docker/nginx/certs/`.
- DNS for `iwerp.com` and host firewall/LB rules are still external.

## SQL Decision Logs

The backend now emits structured `sql_decision_event` lines to container stdout for SQL generation and refusal observability.

Primary stages:

- `report_family_inference`
- `module_inference`
- `shape_support`
- `candidate_selected`
- `refusal`
- `final_verifier`

Reason-code categories include:

- `SQL_REPORT_FAMILY_INFERRED`
- `SQL_REPORT_FAMILY_UNRECOGNIZED`
- `SQL_MODULE_ROUTER_SELECTED`
- `SQL_MODULE_HINT_OVERRIDE`
- `SQL_MODULE_ALIGNMENT_OVERRIDE`
- `SQL_SHAPE_SUPPORTED`
- `SQL_SHAPE_UNSUPPORTED_FIELDS`
- `SQL_SHAPE_UNSUPPORTED_FILTERS`
- `SQL_SHAPE_UNSUPPORTED_ORDERING`
- `SQL_SHAPE_UNSUPPORTED_CALCULATIONS`
- `SQL_REFUSAL_UNSAFE_REQUEST`
- `SQL_REFUSAL_UNSUPPORTED_FIELDS`
- `SQL_REFUSAL_UNSUPPORTED_FILTERS`
- `SQL_REFUSAL_UNSUPPORTED_ORDERING`
- `SQL_REFUSAL_UNSUPPORTED_CALCULATIONS`
- `SQL_REFUSAL_NO_GROUNDED_PATTERN`
- `SQL_REFUSAL_MODULE_ALIGNMENT_FAILED`
- `SQL_REFUSAL_SPECIALIZED_VERIFIER_FAILED`
- `SQL_VERIFIER_PASSED`
- `SQL_VERIFIER_MODULE_ALIGNMENT_FAILED`
- `SQL_VERIFIER_REQUIRED_FIELDS_MISSING`
- `SQL_VERIFIER_REQUIRED_JOINS_MISSING`
- `SQL_VERIFIER_FILTER_FAILED`
- `SQL_VERIFIER_ORDERING_FAILED`
- `SQL_VERIFIER_STYLE_FAILED`

The logs do not include raw user prompts or generated SQL. Instead they include a short query fingerprint, request-shape counts, inferred module/report family, refusal/verifier reason codes, and the selected SQL generation path.

View them in Docker with:

```bash
docker compose logs -f backend
docker compose logs -f backend | grep sql_decision_event
```

Example lines:

```text
2026-04-06 19:24:18 [info     ] sql_decision_event stage=shape_support reason_code=SQL_SHAPE_SUPPORTED trace_id=7cbb... query_fingerprint=8e7f65d6d2a1 query_length=278 task_type=sql_generation router_module=Payables router_module_family=Financials effective_module=Payables module_hint=Payables module_alignment_target=Financials report_family=payables_invoice_details required_field_count=9 required_filter_count=1 required_ordering_count=1 required_calculation_count=0 required_table_count=4 needs_join=True shape_supported=True unsupported_field_count=0 unsupported_filter_count=0 unsupported_ordering_count=0 unsupported_calculation_count=0
2026-04-06 19:24:18 [info     ] sql_decision_event stage=refusal reason_code=SQL_REFUSAL_UNSUPPORTED_FIELDS trace_id=9a21... query_fingerprint=39f1e7b0ca42 query_length=231 task_type=sql_generation router_module=Payables router_module_family=Financials effective_module=Payables module_hint=Payables module_alignment_target=Financials selection_path=none verification_status=FAILED_SQL_NO_GROUNDED_PATTERN verifier_reason_code=SQL_VERIFIER_FAILED_OTHER report_family=payables_payments required_field_count=8 required_filter_count=1 required_ordering_count=0 required_calculation_count=0 required_table_count=4 needs_join=True shape_supported=False unsupported_field_count=1 unsupported_filter_count=0 unsupported_ordering_count=0 unsupported_calculation_count=0 unsupported_fields=['IBAN Number']
```
