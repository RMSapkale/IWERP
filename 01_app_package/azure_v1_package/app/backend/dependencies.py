from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database.session import AsyncSessionLocal
from core.database.models import Tenant, TenantApiKey
from core.security.auth import TenantSecurity, bearer_scheme, api_key_header
import structlog
import datetime

logger = structlog.get_logger(__name__)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_current_tenant(
    auth: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    api_key: str = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """
    Resolves tenant identity via JWT (tenant_id claim) or X-API-Key.
    Returns the full Tenant database object.
    """
    tenant_id = None

    # 1. Try JWT
    if auth:
        payload = TenantSecurity.decode_token(auth)
        tenant_id = payload.get("tenant_id")
    
    # 2. Try API Key if no JWT
    elif api_key:
        api_key_parts = api_key.split("_")
        if len(api_key_parts) < 4: # iwerp_live_{prefix}_{suffix}
            raise HTTPException(status_code=401, detail="Invalid API Key format")
        
        prefix = api_key_parts[2]
        
        # O(1) Lookup by indexed prefix
        result = await db.execute(
            select(TenantApiKey).where(
                TenantApiKey.prefix == prefix,
                TenantApiKey.is_active == True
            )
        )
        candidate_keys = result.scalars().all()
        
        for k in candidate_keys:
            if TenantSecurity.verify_api_key(api_key, k.key_hash):
                tenant_id = k.tenant_id
                # Update last used timestamp
                k.last_used_at = datetime.datetime.now(datetime.timezone.utc)
                await db.commit()
                break
        
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    if not tenant_id:
        raise HTTPException(
            status_code=401, 
            detail="Missing identity. Provide Bearer JWT or X-API-Key."
        )

    # Resolve full tenant object
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
        
    if not tenant.is_active:
        raise HTTPException(status_code=403, detail="Tenant is deactivated")
        
    return tenant
