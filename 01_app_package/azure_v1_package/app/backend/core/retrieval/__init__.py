"""
Hybrid retrieval system for Oracle Fusion SLM.
Includes exact match (FTS) and dense search (pgvector) in Postgres.
"""
from .fts_pg import PostgresFTS
from .router import TaskRouter
from .schema_index import SchemaIndex
from .policy import RetrievalPolicy, RetrievalPlan, RetrievalBudget

try:
    from .vector_pg import PostgresVectorSearch
except ModuleNotFoundError:
    PostgresVectorSearch = None

try:
    from .hybrid import HybridPostgresSearch
except ModuleNotFoundError:
    HybridPostgresSearch = None

try:
    from .reranker import LocalReranker
except ModuleNotFoundError:
    LocalReranker = None

__all__ = [
    "PostgresFTS",
    "PostgresVectorSearch",
    "HybridPostgresSearch",
    "LocalReranker",
    "TaskRouter",
    "SchemaIndex",
    "RetrievalPolicy",
    "RetrievalPlan",
    "RetrievalBudget",
]
