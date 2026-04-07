import contextvars
from typing import Optional
from core.schemas.config import TenantSettings

# Context variable to hold tenant configuration during the lifecycle of a request
_tenant_context: contextvars.ContextVar[Optional[TenantSettings]] = contextvars.ContextVar(
    "tenant_context", default=None
)

class TenantContext:
    """
    Utility class to manage tenant context globally in an async-safe manner.
    Every request MUST have a tenant context set.
    """
    
    @classmethod
    def get(cls) -> TenantSettings:
        tenant = _tenant_context.get()
        if not tenant:
            raise RuntimeError("Tenant context is not set. Isolation breach or improper dependency injection.")
        return tenant
        
    @classmethod
    def set(cls, tenant: TenantSettings) -> contextvars.Token:
        return _tenant_context.set(tenant)
        
    @classmethod
    def reset(cls, token: contextvars.Token) -> None:
        _tenant_context.reset(token)

def get_tenant_context() -> TenantSettings:
    """
    Convenience function to retrieve the current tenant settings from the context.
    """
    return TenantContext.get()

def set_tenant_context(tenant: TenantSettings) -> contextvars.Token:
    """
    Sets the current tenant context. Should only be called by the auth dependency.
    """
    return TenantContext.set(tenant)
