from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class Document(BaseModel):
    document_id: str
    source_uri: str
    title: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    tenant_id: str
    file_paths: List[str]
    job_id: Optional[str] = None

class IngestResponse(BaseModel):
    job_id: str
    status: str
    documents_processed: int
    chunks_created: int
