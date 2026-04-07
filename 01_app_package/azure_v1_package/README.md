# IWERP V1 Azure Deployment Package

This package deploys the Oracle Fusion specialized SLM with sovereign API endpoints:

- `POST /v1/sovereign/chat/completions`
- `POST /v1/sovereign/responses`

Legacy compatibility endpoints remain hidden and are not part of public contract.

## Deployment URL Defaults

- Frontend URL: `https://iwerp.com`
- Backend URL: `https://iwerp.com/api`
- Backend CORS origin: `https://iwerp.com`

## Package Layout

- `app/backend/` full backend application source
- `app/frontend/` full frontend application source
- `backend/` backend deployment assets and startup scripts
- `frontend/` frontend deployment assets
- `azure/` Azure CLI deployment scripts and App Service startup scripts
- `checks/` smoke tests and prompt checklist
- `DEPLOYMENT_GAP_LIST.md` deployment blockers/watchlist from audit
- `AZURE_RELEASE_CHECKLIST.md` release-time verification checklist
- `package_manifest.json` file inventory

The package is standalone. It no longer depends on a separate repo checkout for the app source.

## Recommended V1 Architecture

- Frontend: Azure Static Web Apps (or Azure App Service for Node static)
- Backend API: Azure App Service (Linux, Python)
- Model Runtime: dedicated Azure VM/container host exposing `LOCAL_SLM_BASE_URL`
- Data: Azure Database for PostgreSQL + Azure Files mount for corpora/indexes
- Secrets: Azure Key Vault / App Service App Settings

See `architecture/azure_target_architecture.md` for details.

## Backend Source Layout

```bash
ls app/backend
ls app/backend/core/moe
ls app/backend/moa_routes.py
```

## Frontend Quick Start

```bash
cd app/frontend
npm ci
cp ../../frontend/.env.production.example .env.production
npm run build
npm run preview -- --host 0.0.0.0 --port 3001
```

## Frontend Source Layout

```bash
ls app/frontend/src
ls app/frontend/src/pages
```

## Runtime Entrypoints

```bash
# Backend
./backend/startup.sh

# Frontend App Service sample
./azure/appservice_frontend_startup.sh
```

## Legacy Repo-Relative Instructions Removed

The package-local deployment paths are now:

```bash
# Backend startup
azure/appservice_backend_startup.sh

# Frontend startup
azure/appservice_frontend_startup.sh
```

The package includes:

- backend runtime source
- frontend runtime source
- `moa_routes.py`
- `core/moe/agent_orchestrator.py`
- `core/moe/layer.py`
- `core/moe/prototype.py`

## Backend Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-v1.txt
cp backend/.env.example backend/.env
export APP_ROOT="$(pwd)/app"
./backend/startup.sh
```

## Sovereign Contract Verification

```bash
curl -i https://iwerp.com/api/v1/sovereign/chat/completions
curl -i https://iwerp.com/api/v1/sovereign/responses
```

Both should respond (401 without auth is expected).

## Validation Evidence Included

- `checks/ui_deployed_validation_summary.json` (latest deployed-style local run summary)
- `checks/smoke_test_api.sh` (post-deploy API smoke script)
- `checks/smoke_test_prompts.md` (prompt set for UI/manual verification)

## Latest Local Validation Snapshot

- Source run: `deploy/v1_ui_local_validation_report.json`
- Generated at: `2026-04-06T07:58:06.873043+00:00`
- Result: `21/21` prompts passed
- Contract notes:
  - `what is EPM?`, `what is GL?`, and `what is RMCS?` returned grounded summary answers with citations.
  - `what is payroll gratuity?` is treated as a guarded concept prompt; clean refusal is acceptable if exact concept grounding is absent.
  - Full-shape AP invoice distribution SQL either returns complete grounded SQL or a safe refusal; reduced-shape SQL is not accepted.
  - Guarded refusal outcomes are recorded in `checks/ui_deployed_validation_summary.json` under `accepted_guarded_refusals`.
