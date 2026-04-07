import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-for-dev-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class TenantSecurity:
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict:
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            tenant_id: str = payload.get("tenant_id")
            if tenant_id is None:
                raise HTTPException(status_code=401, detail="Invalid token: missing tenant_id")
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")

    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        return pwd_context.verify(api_key, hashed_key)

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return pwd_context.hash(api_key)

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        return pwd_context.verify(password, hashed_password)
