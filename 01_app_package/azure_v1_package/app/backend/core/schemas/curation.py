from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CorpusType(str, Enum):
    SCHEMA = "schema_corpus"
    SQL = "sql_corpus"
    DOCS = "docs_corpus"
    TROUBLESHOOTING = "troubleshooting_corpus"
    SQL_EXAMPLES = "sql_examples_corpus"
    SCHEMA_METADATA = "schema_metadata_corpus"
    FAST_FORMULA = "fast_formula_corpus"
    REJECT = "reject_corpus"


class DocType(str, Enum):
    SCHEMA_TABLE = "schema_table"
    SCHEMA_VIEW = "schema_view"
    SCHEMA_RELATION = "schema_relation"
    VERIFIED_SQL = "verified_sql"
    SQL_EXAMPLE = "sql_example"
    SCHEMA_METADATA = "schema_metadata"
    FUNCTIONAL_DOC = "functional_doc"
    PROCEDURE_DOC = "procedure_doc"
    NAVIGATION_DOC = "navigation_doc"
    TROUBLESHOOTING_DOC = "troubleshooting_doc"
    SETUP_DOC = "setup_doc"
    FAST_FORMULA_EXAMPLE = "fast_formula_example"
    FAST_FORMULA_DOC = "fast_formula_doc"
    EBS_LEGACY = "ebs_legacy"
    BENCHMARK = "benchmark"
    GENERIC = "generic"


class SourceSystem(str, Enum):
    REPO = "repo"
    GDRIVE = "gdrive"
    METADATA = "metadata"
    BENCHMARK = "benchmark"
    ORACLE_DOCS = "oracle_docs"
    ORACLEWINGS_REPO = "oraclewings_repo"


class AuthorityTier(str, Enum):
    OFFICIAL = "official"
    SECONDARY = "secondary"


class CuratedDocument(BaseModel):
    document_id: str
    source_path: str
    source_uri: str
    title: str
    module: str
    task_type: str
    doc_type: DocType
    trusted_schema_objects: List[str] = Field(default_factory=list)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    source_system: SourceSystem
    corpus: CorpusType
    content: str
    content_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CuratedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    source_path: str
    title: str
    module: str
    task_type: str
    doc_type: DocType
    trusted_schema_objects: List[str] = Field(default_factory=list)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    content_hash: str
    source_system: SourceSystem
    corpus: CorpusType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionManifestRecord(BaseModel):
    source_path: str
    source_uri: Optional[str] = None
    title: str
    module: str
    task_type: str
    doc_type: DocType
    trusted_schema_objects: List[str] = Field(default_factory=list)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    chunk_id: Optional[str] = None
    content_hash: str
    source_system: SourceSystem
    corpus: Optional[CorpusType] = None
    content: Optional[str] = None
    canonical_uri: Optional[str] = None
    doc_release: Optional[str] = None
    authority_tier: Optional[AuthorityTier] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_metadata(self) -> "IngestionManifestRecord":
        if not self.source_path.strip():
            raise ValueError("source_path is required")
        if not self.title.strip():
            raise ValueError("title is required")
        if not self.module.strip():
            raise ValueError("module is required")
        if not self.task_type.strip():
            raise ValueError("task_type is required")
        if not self.content_hash.strip():
            raise ValueError("content_hash is required")
        return self


class RegistryObjectType(str, Enum):
    TABLE = "table"
    VIEW = "view"
    COLUMN = "column"


class TrustedObjectEntry(BaseModel):
    object_name: str
    object_type: RegistryObjectType
    owning_module: str
    original_owning_module: str = "UNKNOWN"
    owning_module_family: str = "UNKNOWN"
    inferred_module: str = "UNKNOWN"
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    inference_source: str = "none"
    low_confidence: bool = True
    manual_lock: bool = False
    aliases: List[str] = Field(default_factory=list)
    ebs_aliases: List[str] = Field(default_factory=list)
    approved_relations: List[str] = Field(default_factory=list)
    relation_details: List[Dict[str, Any]] = Field(default_factory=list)
    source_of_truth: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    columns: List[str] = Field(default_factory=list)
    primary_keys: List[str] = Field(default_factory=list)
    base_tables: List[str] = Field(default_factory=list)
