from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class AppSettings(BaseModel):
    name: str = "oracle-fusion-slm"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "production"

class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit_per_minute: Optional[int] = None

class LlmSettings(BaseModel):
    base_url: str = "http://127.0.0.1:8080"
    model_name: str = "llama-3-8b-instruct"
    max_tokens: int = 4096
    temperature: float = 0.1

class RetrievalSettings(BaseModel):
    vector_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10
    alpha: float = 0.5

class SecuritySettings(BaseModel):
    jwt_secret: str = "CHANGE_ME"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    allowed_api_keys: List[str] = []

class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "json"

class PathsSettings(BaseModel):
    raw_dir: str
    processed_dir: str
    indexes_dir: str
    eval_dir: str
    log_dir: str

class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    api: ApiSettings = ApiSettings()
    llm: LlmSettings = LlmSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()

class TenantSettings(BaseSettings):
    tenant_id: str
    tenant_name: str
    api: Optional[ApiSettings] = None
    retrieval: Optional[RetrievalSettings] = None
    paths: PathsSettings
    security: Optional[SecuritySettings] = None
