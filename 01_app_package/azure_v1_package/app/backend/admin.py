from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from core.database.session import get_db
from core.database.models import Tenant
from pydantic import BaseModel
import secrets
from core.security.auth import TenantSecurity
import structlog

logger = structlog.get_logger(__name__)
admin_router = APIRouter()

class TenantCreate(BaseModel):
    id: str  # e.g. "acme"
    display_name: str

class TenantResponse(BaseModel):
    id: str
    display_name: str
    api_key: str = None
    is_active: bool

async def initialize_tenant_schema(db: AsyncSession, tenant_id: str):
    """
    Automates CREATE SCHEMA and CREATE TABLE for a new tenant.
    """
    schema_name = f"tenant_{tenant_id}"
    try:
        # 1. Create Schema
        await db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))
        
        # 2. Create Tables (Documents, Chunks, Ingest Jobs)
        # We reuse the definitions from init_db.sql but target the new schema
        queries = [
            f"CREATE TABLE IF NOT EXISTS {schema_name}.documents (id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), tenant_id TEXT, filename TEXT NOT NULL, metadata_json JSONB DEFAULT '{{}}'::jsonb, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);",
            f"CREATE TABLE IF NOT EXISTS {schema_name}.chunks (id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), document_id UUID REFERENCES {schema_name}.documents(id) ON DELETE CASCADE, tenant_id TEXT, content TEXT NOT NULL, embedding vector(384), content_tsvector tsvector, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);",
            f"CREATE TABLE IF NOT EXISTS {schema_name}.ingest_jobs (id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), tenant_id TEXT, status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')), filename TEXT NOT NULL, error_message TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);",
            f"CREATE INDEX IF NOT EXISTS idx_{tenant_id}_chunks_fts ON {schema_name}.chunks USING gin (content_tsvector);",
            f"CREATE INDEX IF NOT EXISTS idx_{tenant_id}_chunks_embedding ON {schema_name}.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        ]
        
        for q in queries:
            await db.execute(text(q))
            
        # 3. Add FTS Trigger
        trigger_func = f"""
        CREATE OR REPLACE FUNCTION {schema_name}.chunks_fts_trigger() RETURNS trigger AS $$
        begin
          new.content_tsvector := to_tsvector('english', new.content);
          return new;
        end
        $$ LANGUAGE plpgsql;
        """
        await db.execute(text(trigger_func))
        await db.execute(text(f"DROP TRIGGER IF EXISTS trg_chunks_fts ON {schema_name}.chunks;"))
        await db.execute(text(f"CREATE TRIGGER trg_chunks_fts BEFORE INSERT OR UPDATE ON {schema_name}.chunks FOR EACH ROW EXECUTE FUNCTION {schema_name}.chunks_fts_trigger();"))
        
        await db.commit()
        logger.info("tenant_schema_initialized", tenant_id=tenant_id, schema=schema_name)
    except Exception as e:
        await db.rollback()
        logger.error("tenant_schema_initialization_failed", tenant_id=tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to initialize tenant schema: {str(e)}")

@admin_router.post("/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(payload: TenantCreate, db: AsyncSession = Depends(get_db)):
    # Check if exists
    existing = await db.execute(select(Tenant).where(Tenant.id == payload.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Tenant ID already exists")

    # Generate API Key
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    api_key_hash = TenantSecurity.hash_api_key(api_key)
    
    new_tenant = Tenant(
        id=payload.id,
        display_name=payload.display_name,
        api_key_hash=api_key_hash,
        is_active=True
    )
    db.add(new_tenant)
    await db.commit()
    
    # Initialize Schema
    await initialize_tenant_schema(db, payload.id)
    
    return TenantResponse(
        id=new_tenant.id,
        display_name=new_tenant.display_name,
        api_key=api_key,
        is_active=True
    )

@admin_router.post("/tenants/{tenant_id}/rotate-keys", response_model=TenantResponse)
async def rotate_tenant_key(tenant_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
        
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    tenant.api_key_hash = TenantSecurity.hash_api_key(api_key)
    await db.commit()
    
    return TenantResponse(
        id=tenant.id,
        display_name=tenant.display_name,
        api_key=api_key,
        is_active=tenant.is_active
    )

@admin_router.delete("/tenants/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(tenant_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
        
    tenant.is_active = False # Soft delete
    await db.commit()
    return None
