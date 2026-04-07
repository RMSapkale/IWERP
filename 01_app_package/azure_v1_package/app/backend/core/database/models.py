from sqlalchemy import Column, String, Integer, Text, ForeignKey, JSON, DateTime, func, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import relationship
import uuid
import datetime
from .session import Base

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True)  # e.g. "demo"
    display_name = Column(String)
    is_active = Column(Boolean, default=True) # Soft delete
    password_hash = Column(String, nullable=True) # For UI login
    moe_enabled = Column(Boolean, default=False)
    moe_experiment_group = Column(String, default="control") # "control" or "treatment"
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    api_keys = relationship("TenantApiKey", back_populates="tenant", cascade="all, delete-orphan")

class TenantApiKey(Base):
    __tablename__ = "tenant_api_keys"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    name = Column(String, nullable=False, default="Default Key")
    prefix = Column(String, index=True, nullable=True) # First 8 chars of key for O(1) lookup
    key_hash = Column(String, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="api_keys")

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    filename = Column(String, nullable=False)
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    tenant_id = Column(String, index=True, nullable=False) # For efficient isolation
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True) # Changed from Vector(384) to JSON to bypass pgvector requirement
    
    # Custom TSVector for Full-Text Search
    content_tsvector = Column(TSVECTOR)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    document = relationship("Document", back_populates="chunks")

class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, index=True, nullable=False)
    status = Column(String, nullable=False, default="pending") # pending, processing, completed, failed
    filename = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SFTSample(Base):
    __tablename__ = "sft_samples"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, index=True, nullable=False)
    module = Column(String, nullable=False) # AP, AR, GL, etc.
    task_type = Column(String, nullable=False) # fusion_nav, fusion_proc, etc.
    messages = Column(JSON, nullable=False) # List of {"role": "...", "content": "..."}
    difficulty = Column(String, default="medium") # easy, medium, hard
    source_doc_ids = Column(JSON, default=[]) # List of document UUIDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TraceRecord(Base):
    __tablename__ = "trace_records"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, index=True, nullable=False)
    query = Column(Text, nullable=False)
    task_type = Column(String, nullable=True)
    module = Column(String, nullable=True)
    retrieved_chunk_ids = Column(JSON, default=[]) # List of chunk UUIDs
    retrieval_scores = Column(JSON, default={}) # {uuid: score}
    final_context_size = Column(Integer)
    model_settings = Column(JSON, default={})
    latency_ms = Column(JSON, default={}) # {step: ms}
    audit_logs = Column(JSON, default={}) # {hallucination_score: ..., etc.}
    policy_decisions = Column(JSON, default={}) # {decision: reason}
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("trace_records.id"), nullable=False)
    tenant_id = Column(String, index=True, nullable=False)
    rating = Column(Integer) # 1 for thumbs up, -1 for thumbs down
    issue_type = Column(String, nullable=True) # hallucination, bad_retrieval, etc.
    comment = Column(Text, nullable=True)
    corrected_answer = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Indexes
# (Custom RAG indexes removed in favor of FAISS)
