# Deployment Gap List (V1)

Date: 2026-04-06

## Blocking Gaps

None identified for deployability of the current sovereign UI/API contract.

Standalone package note:
- The deployment package now includes full app source under `app/backend` and `app/frontend`.
- MOA/MOE runtime files are included in-package and do not require a separate repo checkout.

## Watchlist Gaps

1. Guarded refusals are intentional on weakly grounded prompts
- Evidence: `deploy/v1_ui_local_validation_report.json` and `checks/ui_deployed_validation_summary.json`
- Current state: latest local validation passed `21/21`, including accepted guarded refusals for:
  - payroll gratuity summary
  - payroll gratuity Fast Formula generation
  - full-shape AP invoice distribution SQL
- Action: keep these prompts in smoke coverage so the product never degrades into weak procedural substitutions or reduced-shape SQL.

2. Storage mount assumptions must be explicit in Azure rollout
- Corpora/index artifacts are expected under `IWERP_BASE_DIR`.
- Action: ensure Azure Files (or equivalent) mount is available and path-aligned before cutover.

3. `/v1/health/readiness` currently auth-protected
- Operational tooling must use authenticated readiness checks (or rely on `/health` for unauth probe).
- Action: keep as-is for v1 safety; document in runbook.

4. Frontend App Service startup path currently builds at boot in sample script
- Slower cold start if used directly (`npm ci && npm run build`).
- Action: prefer prebuilt static deployment (Static Web Apps) or containerized NGINX image.
