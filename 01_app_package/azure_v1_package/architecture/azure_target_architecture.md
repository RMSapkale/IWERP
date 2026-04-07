# Azure Target Architecture (V1)

## Goals

- Preserve certified runtime behavior
- Keep sovereign endpoint contract
- Keep deployment simple and operable

## Components

1. Frontend hosting  
- Azure Static Web Apps (preferred) OR Azure App Service (Node)  
- Hosts built React/Vite UI  
- `VITE_API_BASE_URL` points to `https://iwerp.com/api`

2. Backend API hosting  
- Azure App Service Linux (Python 3.11/3.12 recommended)  
- Runs `uvicorn backend.main:app` via startup script  
- Exposes `/health` and `/v1/sovereign/*`

3. Inference runtime  
- Separate Azure VM/container host running local SLM endpoint  
- Backend uses `LOCAL_SLM_BASE_URL` (example: `http://<model-host>:8080`)

4. Persistence  
- Azure Database for PostgreSQL (tenant/auth + metadata tables)  
- Azure Files mount for corpora/indexes/artifacts:
  - `backend/core/retrieval/vectors/faiss/...`
  - `specialization_tracks/...`
  - `grounding_answerability_pass/...`
  - `coverage_expansion/...`

5. Secrets/config  
- App Service Application Settings (or Key Vault references)
- Do not hardcode secrets in code/package

6. Monitoring  
- App Service logs + Azure Monitor alerts
- Health probe on `/health`

## Network/Access

- Public: frontend URL `https://iwerp.com`, backend URL `https://iwerp.com/api`
- Restricted/private: model host endpoint and storage mounts
- CORS: set `ALLOWED_ORIGINS=https://iwerp.com`

## Why this shape

- Avoids architecture redesign
- Keeps model runtime independently scalable
- Supports existing backend behavior with minimal change
