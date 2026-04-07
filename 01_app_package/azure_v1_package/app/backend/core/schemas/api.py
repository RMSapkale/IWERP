from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: Role
    content: str

class Citation(BaseModel):
    citation_id: str
    document_id: str
    title: str
    snippet: str
    module: str
    source: str
    corpus: str
    score: float
    source_uri: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.10

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: Optional[str] = None

class ChatResponse(BaseModel):
    id: str = Field(..., description="Trace ID")
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    citations: List[Citation] = Field(default_factory=list)
    usage: Dict[str, int] = Field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)
    audit: Dict[str, Any] = Field(default_factory=dict)

class UserIntent(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    CODE_GEN = "code_gen"
    TABLE_LOOKUP = "table_lookup"
    SQL_GENERATION = "sql_generation"
    SQL_TROUBLESHOOTING = "sql_troubleshooting"
    TROUBLESHOOTING = "troubleshooting"
    FAST_FORMULA_GENERATION = "fast_formula_generation"
    FAST_FORMULA_TROUBLESHOOTING = "fast_formula_troubleshooting"
    NAVIGATION = "navigation"
    PROCEDURE = "procedure"
    INTEGRATION = "integration"
    REPORT_LOGIC = "report_logic"
    FUSION_NAV = "fusion_nav" # Mapping to old for compatibility
    FUSION_PROC = "fusion_proc"
    FUSION_TROUBLESHOOT = "fusion_troubleshoot"
    FUSION_INTEGRATION = "fusion_integration"
    UNKNOWN = "unknown"

class VerificationTag(str, Enum):
    CONFIRMED_FUSION = "confirmed_fusion"
    MAPPED_FROM_EBS = "mapped_from_ebs"
    UNCONFIRMED = "unconfirmed"

class AuditScore(BaseModel):
    hallucination_score: float = Field(0.0, ge=0.0, le=1.0)
    sql_validity_score: float = Field(0.0, ge=0.0, le=1.0)
    module_accuracy_score: float = Field(0.0, ge=0.0, le=1.0)
    grounding_score: float = Field(0.0, ge=0.0, le=1.0)
    verbosity_score: float = Field(0.0, ge=0.0, le=1.0)
    detected_tables: List[str] = Field(default_factory=list)
    mapped_tables: List[str] = Field(default_factory=list)
    rejected_tables: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None
