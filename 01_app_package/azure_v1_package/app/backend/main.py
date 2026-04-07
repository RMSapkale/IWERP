import os
from fastapi.responses import RedirectResponse, FileResponse
from .routes import router
from .admin import admin_router
from .auth_routes import auth_router
from .moa_routes import moa_router
from .embeddings_routes import router as embeddings_router
from .ops_routes import legacy_openai_router, ops_router

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="IWERP AI — Production SLM API",
    description="Production-grade multi-tenant SLM with unique API key generation",
    version="1.0.0"
)
print("IWERP backend startup")

allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


enable_legacy_openai_routes = _env_flag_enabled("IWFUSION_ENABLE_LEGACY_OPENAI_ROUTES", default=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/v1")
app.include_router(ops_router, prefix="/v1")
if enable_legacy_openai_routes:
    app.include_router(legacy_openai_router, prefix="/v1")
app.include_router(admin_router, prefix="/v1/admin", tags=["admin"])
app.include_router(auth_router, prefix="/v1/auth", tags=["auth"])
app.include_router(moa_router, prefix="/v2", tags=["moa"])
app.include_router(embeddings_router, prefix="/v1", tags=["sovereign-rag"])

# Use relative path to frontend/dist within the iwerp-prod bundle
DIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "postgres"}

if os.path.exists(DIST_PATH):
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # 1. Try to serve exact file from DIST_PATH
        file_path = os.path.join(DIST_PATH, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # 2. If it's an API route that somehow reached here, let it 404
        if full_path.startswith("v1/") or full_path.startswith("v2/") or full_path == "health":
            raise HTTPException(status_code=404)
            
        # 3. Fallback to index.html for React Router deep links
        return FileResponse(os.path.join(DIST_PATH, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
