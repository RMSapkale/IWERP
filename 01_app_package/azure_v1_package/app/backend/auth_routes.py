from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Dict, Any
import secrets
import uuid
import datetime
import structlog

from core.database.session import get_db
from core.database.models import Tenant, TenantApiKey
from core.security.auth import TenantSecurity

from core.security.auth import TenantSecurity

logger = structlog.get_logger(__name__)

auth_router = APIRouter()

class LoginRequest(BaseModel):
    # Accept both field names — frontend sends 'username', internal uses 'tenant_id'
    username: str | None = None
    tenant_id: str | None = None
    password: str

    def get_tenant_id(self) -> str:
        # username maps to tenant_id (lowercased, underscored)
        raw = self.username or self.tenant_id or ""
        return raw.lower().replace(" ", "_")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class KeyResponse(BaseModel):
    id: str
    name: str
    api_key: str
    created_at: Any

class KeyListItem(BaseModel):
    id: str
    name: str
    api_key_masked: str
    created_at: Any
    is_active: bool

class RegisterRequest(BaseModel):
    username: str
    password: str
    tenant_name: str

@auth_router.post("/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    tenant_id = payload.tenant_name.lower().replace(" ", "_")
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Tenant already exists")

    hashed_pw = TenantSecurity.hash_password(payload.password)
    tenant = Tenant(
        id=tenant_id,
        display_name=payload.tenant_name,
        password_hash=hashed_pw,
        is_active=True
    )
    db.add(tenant)
    await db.commit()

    token = TenantSecurity.create_access_token({"tenant_id": tenant_id})
    return {"access_token": token, "token_type": "bearer"}


@auth_router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Tenant).where(Tenant.id == payload.get_tenant_id()))
    tenant = result.scalar_one_or_none()
    
    if not tenant or not tenant.password_hash:
        raise HTTPException(status_code=401, detail="Invalid tenant or password not set")
    
    if not TenantSecurity.verify_password(payload.password, tenant.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")
    
    token = TenantSecurity.create_access_token({"tenant_id": tenant.id})
    return {"access_token": token, "token_type": "bearer"}

@auth_router.get("/keys/list", response_model=list[KeyListItem])
async def list_keys(
    db: AsyncSession = Depends(get_db),
    token_payload: dict = Depends(TenantSecurity.decode_token)
):
    tenant_id = token_payload.get("tenant_id")
    result = await db.execute(
        select(TenantApiKey).where(TenantApiKey.tenant_id == tenant_id)
    )
    keys = result.scalars().all()
    return [
        {
            "id": str(k.id),
            "name": k.name,
            "api_key_masked": f"iwerp_live_{k.prefix or '...'}{'****'}",
            "created_at": k.created_at,
            "is_active": k.is_active,
            "last_used_at": k.last_used_at
        } 
        for k in keys
    ]

class CreateKeyRequest(BaseModel):
    name: str | None = None

@auth_router.post("/keys/create", response_model=KeyResponse)
@auth_router.post("/keys/rotate", response_model=KeyResponse) # Deprecated but kept for compat
async def create_key(
    payload: CreateKeyRequest = None,
    db: AsyncSession = Depends(get_db),
    token_payload: dict = Depends(TenantSecurity.decode_token)
):
    tenant_id = token_payload.get("tenant_id")
    
    # Generate new unique key
    suffix = secrets.token_urlsafe(32)
    prefix = secrets.token_urlsafe(8)[:8] # 8 chars prefix
    raw_key = f"iwerp_live_{prefix}_{suffix}"
    key_hash = TenantSecurity.hash_api_key(raw_key)
    
    name = (payload.name if payload else None) or f"Key generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

    new_key = TenantApiKey(
        tenant_id=tenant_id,
        name=name,
        prefix=prefix,
        key_hash=key_hash
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)
    
    return {
        "id": str(new_key.id),
        "name": new_key.name,
        "api_key": raw_key,
        "created_at": new_key.created_at
    }

@auth_router.delete("/keys/revoke/{key_id}")
async def revoke_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    token_payload: dict = Depends(TenantSecurity.decode_token)
):
    tenant_id = token_payload.get("tenant_id")
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid key ID format")
        
    result = await db.execute(
        select(TenantApiKey).where(TenantApiKey.id == key_uuid, TenantApiKey.tenant_id == tenant_id)
    )
    key = result.scalar_one_or_none()
    if not key:
        logger.warning("revoke_key_not_found", key_id=key_id, tenant_id=tenant_id)
        raise HTTPException(status_code=404, detail="Key not found")
        
    logger.info("revoking_key_success", key_id=key_id, tenant_id=tenant_id)
    await db.delete(key)
    await db.commit()
    return {"status": "success", "message": "Key revoked"}
