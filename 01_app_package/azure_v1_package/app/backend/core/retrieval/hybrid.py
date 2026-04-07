import os
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from core.retrieval.fts_pg import PostgresFTS
from core.retrieval.policy import RetrievalPolicy
from core.retrieval.schema_index import SchemaIndex
from core.retrieval.vectors.faiss_index import FaissIndex
from core.schemas.curation import CorpusType
from core.schemas.router import FusionModule, ModuleFamily, TaskType, module_families_for_value


class HybridPostgresSearch:
    """
    Retrieval orchestrator that keeps schema, SQL, and docs corpora separate.
    """

    CURATED_CORPORA = {
        CorpusType.SCHEMA.value,
        CorpusType.SQL.value,
        CorpusType.DOCS.value,
        CorpusType.TROUBLESHOOTING.value,
        CorpusType.SQL_EXAMPLES.value,
        CorpusType.SCHEMA_METADATA.value,
        CorpusType.FAST_FORMULA.value,
    }
    EXACT_MODULE_ALIASES = {
        FusionModule.AP.value: FusionModule.PAYABLES.value,
        FusionModule.AR.value: FusionModule.RECEIVABLES.value,
        FusionModule.GL.value: FusionModule.GENERAL_LEDGER.value,
        "CE": FusionModule.CASH_MANAGEMENT.value,
        "FA": FusionModule.ASSETS.value,
        "EXM": FusionModule.EXPENSES.value,
        "ZX": FusionModule.TAX.value,
    }
    STRICT_FINANCIAL_LEAF_MODULES = {
        FusionModule.PAYABLES.value,
        FusionModule.RECEIVABLES.value,
        FusionModule.GENERAL_LEDGER.value,
        FusionModule.CASH_MANAGEMENT.value,
        FusionModule.ASSETS.value,
        FusionModule.EXPENSES.value,
        FusionModule.TAX.value,
        FusionModule.AP.value,
        FusionModule.AR.value,
        FusionModule.GL.value,
        "CE",
        "FA",
        "EXM",
        "ZX",
    }
    DOC_GROUNDING_CORPORA = {
        CorpusType.DOCS.value,
        CorpusType.TROUBLESHOOTING.value,
    }
    SHARED_CURATED_TENANT_ID = "demo"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enable_fts: bool = False):
        self.enable_fts = enable_fts
        self.fts = PostgresFTS() if enable_fts else None
        self.model_name = model_name
        self.schema_index = SchemaIndex()
        self._faiss_cache: Dict[tuple[str, str], FaissIndex] = {}
        self.base_dir = os.path.join(os.path.dirname(__file__), "vectors")

    def _get_faiss_index(self, tenant_id: str, corpus: str) -> FaissIndex:
        cache_key = (tenant_id, corpus)
        if cache_key not in self._faiss_cache:
            self._faiss_cache[cache_key] = FaissIndex(
                tenant_id=tenant_id,
                indexes_dir=self.base_dir,
                embedding_model=self.model_name,
                corpus=corpus,
            )
        return self._faiss_cache[cache_key]

    def _shared_curated_tenant(self, tenant_id: str, corpus: str) -> Optional[str]:
        if corpus not in self.CURATED_CORPORA:
            return None
        if tenant_id == self.SHARED_CURATED_TENANT_ID:
            return None
        return self.SHARED_CURATED_TENANT_ID

    def _normalize_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = {}
        if not filters:
            return normalized
        for key, value in filters.items():
            if value in (None, "", [], (), {}):
                continue
            normalized[key] = value
        return normalized

    def _corpora(self, filters: Dict[str, Any]) -> List[str]:
        corpora = filters.get("corpora")
        if not corpora:
            corpus = filters.get("corpus")
            if corpus:
                corpora = [corpus] if not isinstance(corpus, (list, tuple, set)) else list(corpus)
        if not corpora:
            return [CorpusType.DOCS.value]
        return [str(corpus) for corpus in corpora if str(corpus) in self.CURATED_CORPORA]

    def _infer_module(self, hit: Dict[str, Any]) -> Optional[str]:
        metadata = hit.get("metadata") or {}
        family = metadata.get("module_family") or metadata.get("module")
        if family:
            normalized = next(iter(module_families_for_value(str(family))), ModuleFamily.UNKNOWN.value)
            if normalized != ModuleFamily.UNKNOWN.value:
                return normalized

        text = " ".join(
            [
                str(hit.get("filename") or ""),
                str(metadata.get("filename") or ""),
                str(metadata.get("title") or ""),
                str(hit.get("content") or "")[:200],
            ]
        ).upper()

        for candidate in re.findall(r"\b[A-Z]{2,}\b", text):
            normalized = next(iter(module_families_for_value(candidate)), ModuleFamily.UNKNOWN.value)
            if normalized != ModuleFamily.UNKNOWN.value:
                return normalized
        return None

    def _module_metadata(self, hit: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        metadata = hit.get("metadata") or {}
        module_value = str(metadata.get("module") or "").strip()
        family_value = str(metadata.get("module_family") or "").strip()
        family_values = set(module_families_for_value(module_value or family_value))

        normalized_family = next(iter(family_values), ModuleFamily.UNKNOWN.value)
        if normalized_family == ModuleFamily.UNKNOWN.value:
            inferred_family = self._infer_module(hit)
            normalized_family = inferred_family or ModuleFamily.UNKNOWN.value

        exact_module = None
        if module_value and module_value not in {family.value for family in ModuleFamily}:
            exact_module = module_value

        return exact_module, normalized_family

    def _canonical_exact_module(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = str(value).strip()
        return self.EXACT_MODULE_ALIASES.get(normalized, normalized)

    def _requested_exact_modules(self, requested_module: Optional[Any]) -> set[str]:
        requested_exact: set[str] = set()
        values = requested_module if isinstance(requested_module, (list, tuple, set)) else [requested_module]
        for value in values:
            if not value:
                continue
            normalized_value = str(value)
            if normalized_value and normalized_value not in {family.value for family in ModuleFamily}:
                canonical = self._canonical_exact_module(normalized_value)
                if canonical:
                    requested_exact.add(canonical)
        return requested_exact

    def _is_strict_financial_leaf_request(self, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters or not filters.get("strict_exact_module_only"):
            return False
        requested_exact = self._requested_exact_modules(filters.get("exact_module_allowlist") or filters.get("requested_module"))
        return bool(requested_exact & self.STRICT_FINANCIAL_LEAF_MODULES)

    def _passes_module_firewall(self, hit: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True

        requested_exact = self._requested_exact_modules(filters.get("exact_module_allowlist") or filters.get("requested_module"))
        if not requested_exact:
            return True

        actual_exact, actual_family = self._module_metadata(hit)
        actual_exact = self._canonical_exact_module(actual_exact)

        if actual_exact in requested_exact:
            return True

        if not filters.get("strict_exact_module_only"):
            return True

        if not filters.get("allow_same_family_fallback"):
            return False

        requested_families = set()
        for exact_module in requested_exact:
            requested_families.update(module_families_for_value(exact_module))
        requested_families.discard(ModuleFamily.UNKNOWN.value)

        if actual_family == ModuleFamily.COMMON.value:
            return False

        return bool(actual_family and actual_family in requested_families)

    def _module_factor(self, hit: Dict[str, Any], requested_module: Optional[Any], filters: Optional[Dict[str, Any]] = None) -> float:
        requested_families: set[str] = set()
        requested_exact = self._requested_exact_modules(requested_module)
        if isinstance(requested_module, (list, tuple, set)):
            for value in requested_module:
                normalized_value = str(value)
                requested_families.update(module_families_for_value(normalized_value))
        elif requested_module:
            normalized_value = str(requested_module)
            requested_families.update(module_families_for_value(normalized_value))

        requested_families.discard(ModuleFamily.UNKNOWN.value)
        requested_families.discard(ModuleFamily.COMMON.value)

        if not requested_families:
            return 1.0

        actual_exact, actual_family = self._module_metadata(hit)
        actual_exact = self._canonical_exact_module(actual_exact)
        actual_corpus = str((hit.get("metadata") or {}).get("corpus") or "")
        actual_doclike = actual_corpus in self.DOC_GROUNDING_CORPORA
        requested_financial_leaf = bool(requested_exact & self.STRICT_FINANCIAL_LEAF_MODULES)
        allow_same_family_fallback = bool(filters and filters.get("allow_same_family_fallback"))
        if requested_exact:
            if actual_exact in requested_exact:
                return 2.35 if actual_doclike else 1.95
            if actual_family == ModuleFamily.COMMON.value:
                return 0.0 if requested_financial_leaf else (0.1 if actual_doclike else 0.18)
            if actual_family in requested_families:
                if requested_financial_leaf and actual_exact and actual_exact not in requested_exact and actual_doclike:
                    return 0.03 if allow_same_family_fallback else 0.0
                if requested_financial_leaf:
                    return 0.03 if allow_same_family_fallback else 0.0
                return 0.22 if actual_doclike else 0.3
            return 0.02 if (requested_financial_leaf and allow_same_family_fallback) else (0.0 if requested_financial_leaf else (0.12 if actual_family and actual_family != ModuleFamily.UNKNOWN.value else 0.25))

        if not actual_family or actual_family == ModuleFamily.UNKNOWN.value:
            return 0.5
        if actual_family == ModuleFamily.COMMON.value and ModuleFamily.COMMON.value not in requested_families:
            return 0.4
        return 1.0 if actual_family in requested_families else 0.18

    def _source_factor(self, hit: Dict[str, Any]) -> float:
        metadata = hit.get("metadata") or {}
        source_system = str(metadata.get("source_system") or "").lower()
        authority_tier = str(metadata.get("authority_tier") or "").lower()
        source_uri = str(metadata.get("source_uri") or metadata.get("source_path") or "").lower()

        factor = 1.0
        if authority_tier == "official":
            factor *= 1.2
        elif authority_tier == "secondary":
            factor *= 0.92

        if source_system == "oracle_docs":
            factor *= 1.12
        elif source_system == "oraclewings_repo":
            factor *= 0.94
        elif source_system == "metadata":
            factor *= 1.05

        if "community_data" in source_uri:
            factor *= 0.7
        return factor

    def _corpus_priority_factor(self, corpus: str, corpora: List[str], requested_task: Optional[Any]) -> float:
        if corpus not in corpora:
            return 1.0
        position = corpora.index(corpus)
        base = {0: 1.25, 1: 1.05, 2: 0.9}.get(position, 0.85)
        task_value = str(requested_task or "")
        if corpus == CorpusType.SQL.value and task_value in {
            TaskType.SQL_GENERATION.value,
            TaskType.SQL_TROUBLESHOOTING.value,
            TaskType.REPORT_LOGIC.value,
            TaskType.TROUBLESHOOTING.value,
        }:
            return base * 1.15
        if corpus == CorpusType.SQL_EXAMPLES.value and task_value in {
            TaskType.SQL_GENERATION.value,
            TaskType.SQL_TROUBLESHOOTING.value,
            TaskType.REPORT_LOGIC.value,
        }:
            return base * 1.2
        if corpus == CorpusType.SCHEMA_METADATA.value and task_value in {
            TaskType.TABLE_LOOKUP.value,
            TaskType.SQL_GENERATION.value,
            TaskType.SQL_TROUBLESHOOTING.value,
            TaskType.REPORT_LOGIC.value,
        }:
            return base * 1.12
        if corpus == CorpusType.TROUBLESHOOTING.value and task_value == TaskType.TROUBLESHOOTING.value:
            return base * 1.35
        if corpus == CorpusType.FAST_FORMULA.value and task_value in {
            TaskType.FAST_FORMULA_GENERATION.value,
            TaskType.FAST_FORMULA_TROUBLESHOOTING.value,
        }:
            return base * 1.2
        if corpus == CorpusType.DOCS.value and task_value in {
            TaskType.PROCEDURE.value,
            TaskType.NAVIGATION.value,
            TaskType.GENERAL.value,
            TaskType.SUMMARY.value,
        }:
            return base * 1.1
        return base

    def _task_factor(self, hit: Dict[str, Any], requested_task: Optional[Any]) -> float:
        if not requested_task:
            return 1.0

        metadata = hit.get("metadata") or {}
        corpus = metadata.get("corpus")
        task_value = str(requested_task)

        if task_value == TaskType.TABLE_LOOKUP.value:
            return 1.3 if corpus == CorpusType.SCHEMA.value else 0.1
        if task_value == TaskType.SQL_GENERATION.value:
            if corpus == CorpusType.SQL.value:
                return 1.35
            if corpus == CorpusType.SQL_EXAMPLES.value:
                return 1.45
            if corpus == CorpusType.SCHEMA_METADATA.value:
                return 1.15
            if corpus == CorpusType.SCHEMA.value:
                return 1.05
            return 0.2
        if task_value == TaskType.SQL_TROUBLESHOOTING.value:
            if corpus == CorpusType.SQL_EXAMPLES.value:
                return 1.4
            if corpus == CorpusType.SCHEMA_METADATA.value:
                return 1.15
            if corpus == CorpusType.DOCS.value:
                return 0.45
            return 0.15
        if task_value == TaskType.REPORT_LOGIC.value:
            if corpus == CorpusType.SQL.value:
                return 1.3
            if corpus == CorpusType.SQL_EXAMPLES.value:
                return 1.35
            if corpus == CorpusType.SCHEMA_METADATA.value:
                return 1.15
            if corpus == CorpusType.SCHEMA.value:
                return 1.05
            return 0.2
        if task_value == TaskType.FAST_FORMULA_GENERATION.value:
            return 1.45 if corpus == CorpusType.FAST_FORMULA.value else 0.1
        if task_value == TaskType.FAST_FORMULA_TROUBLESHOOTING.value:
            return 1.45 if corpus == CorpusType.FAST_FORMULA.value else 0.1
        if task_value == TaskType.TROUBLESHOOTING.value:
            if corpus == CorpusType.TROUBLESHOOTING.value:
                return 1.55
            if corpus == CorpusType.DOCS.value:
                return 1.15
            if corpus == CorpusType.SQL.value:
                return 0.95
            if corpus == CorpusType.SQL_EXAMPLES.value:
                return 0.9
            if corpus == CorpusType.SCHEMA_METADATA.value:
                return 0.7
            if corpus == CorpusType.SCHEMA.value:
                return 0.45
            return 0.2
        if task_value in {TaskType.PROCEDURE.value, TaskType.NAVIGATION.value}:
            return 1.3 if corpus == CorpusType.DOCS.value else 0.05
        if task_value in {TaskType.GENERAL.value, TaskType.SUMMARY.value}:
            if corpus == CorpusType.DOCS.value:
                return 1.15
            if corpus == CorpusType.SCHEMA.value:
                return 0.7
            return 0.2

        return 1.0

    def _normalize_hit(self, hit: Dict[str, Any], default_score_key: str) -> Dict[str, Any]:
        metadata = dict(hit.get("metadata") or {})
        filename = (
            hit.get("filename")
            or metadata.get("filename")
            or metadata.get("title")
            or metadata.get("source_path")
            or metadata.get("source")
            or "grounding-source"
        )
        metadata.setdefault("filename", filename)
        metadata.setdefault("title", metadata.get("title") or filename)
        metadata.setdefault("source_uri", metadata.get("source_uri") or metadata.get("source_path") or filename)

        normalized = dict(hit)
        normalized["metadata"] = metadata
        normalized["filename"] = str(filename)
        normalized["score"] = float(hit.get("score") or hit.get(default_score_key) or 0.0)
        normalized["combined_score"] = float(hit.get("combined_score") or normalized["score"])
        return normalized

    async def _search_docs_or_sql(
        self,
        db: AsyncSession,
        tenant_id: str,
        query: str,
        corpus: str,
        filters: Dict[str, Any],
        fts_limit: int,
        vector_limit: int,
        fts_weight: float,
        vector_weight: float,
    ) -> List[Dict[str, Any]]:
        corpus_filters = dict(filters)
        corpus_filters["corpus"] = corpus
        requested_task = corpus_filters.get("task_type")
        if requested_task:
            try:
                task_enum = TaskType(str(requested_task))
                task_filters = RetrievalPolicy.task_filters_for_corpus(task_enum, corpus)
                if task_filters:
                    corpus_filters["task_type"] = task_filters
            except ValueError:
                pass

        fts_hits = []
        if self.enable_fts and self.fts is not None and db is not None:
            try:
                fts_hits = await self.fts.search(db, tenant_id, query, limit=fts_limit, filters=corpus_filters)
            except Exception:
                fts_hits = []

        vector_hits = self._get_faiss_index(tenant_id, corpus).query(query, top_k=vector_limit, filters=corpus_filters)
        if not vector_hits:
            shared_tenant_id = self._shared_curated_tenant(tenant_id, corpus)
            if shared_tenant_id:
                vector_hits = self._get_faiss_index(shared_tenant_id, corpus).query(
                    query,
                    top_k=vector_limit,
                    filters=corpus_filters,
                )

        merged_hits: Dict[str, Dict[str, Any]] = {}
        for raw_hit in fts_hits:
            hit = self._normalize_hit(raw_hit, "score")
            chunk_id = str(hit["id"])
            merged_hits[chunk_id] = {
                **hit,
                "id": chunk_id,
                "fts_score": hit["score"],
                "vector_score": 0.0,
                "combined_score": hit["score"] * fts_weight,
            }

        for raw_hit in vector_hits:
            hit = self._normalize_hit(raw_hit, "score")
            chunk_id = str(hit.get("chunk_id") or hit.get("id"))
            if chunk_id in merged_hits:
                merged_hits[chunk_id]["vector_score"] = hit["score"]
                merged_hits[chunk_id]["combined_score"] += hit["score"] * vector_weight
                merged_hits[chunk_id]["score"] = merged_hits[chunk_id]["combined_score"]
                continue

            merged_hits[chunk_id] = {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "document_id": hit.get("document_id"),
                "content": hit.get("content", ""),
                "filename": hit["filename"],
                "metadata": hit.get("metadata", {}),
                "fts_score": 0.0,
                "vector_score": hit["score"],
                "combined_score": hit["score"] * vector_weight,
                "score": hit["score"] * vector_weight,
            }

        return list(merged_hits.values())

    async def search(
        self,
        db: AsyncSession,
        tenant_id: str,
        query: str,
        limit: int = 20,
        fts_limit: int = 40,
        vector_limit: int = 40,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filters = self._normalize_filters(filters)
        requested_module = (
            filters.get("requested_module")
            or filters.get("module_family")
            or filters.get("module")
        )
        requested_task = filters.get("task_type")
        corpora = self._corpora(filters)

        all_hits: Dict[str, Dict[str, Any]] = {}

        for corpus in corpora:
            if corpus == CorpusType.SCHEMA.value:
                schema_limit = min(limit, int(filters.get("schema_limit", max(2, limit // 2 or 1))))
                for hit in self.schema_index.search(query, module=requested_module, top_k=schema_limit, filters=filters):
                    normalized = self._normalize_hit(hit, "score")
                    normalized["combined_score"] = float(normalized.get("combined_score") or normalized.get("score") or 0.0)
                    all_hits[str(hit["id"])] = normalized
                continue

            corpus_hits = await self._search_docs_or_sql(
                db=db,
                tenant_id=tenant_id,
                query=query,
                corpus=corpus,
                filters=filters,
                fts_limit=min(fts_limit, limit * 2),
                vector_limit=min(vector_limit, limit * 2),
                fts_weight=fts_weight,
                vector_weight=vector_weight,
            )
            for hit in corpus_hits:
                all_hits[str(hit["id"])] = hit

        ranked_hits = []
        for hit in all_hits.values():
            if not self._passes_module_firewall(hit, filters):
                continue
            hit["combined_score"] = float(hit.get("combined_score") or hit.get("score") or 0.0)
            hit["combined_score"] *= self._module_factor(hit, requested_module, filters)
            hit["combined_score"] *= self._task_factor(hit, requested_task)
            hit["combined_score"] *= self._source_factor(hit)
            hit["combined_score"] *= self._corpus_priority_factor(
                str((hit.get("metadata") or {}).get("corpus") or ""),
                corpora,
                requested_task,
            )
            hit["score"] = hit["combined_score"]
            ranked_hits.append(hit)

        ranked_hits.sort(key=lambda item: item["combined_score"], reverse=True)
        return ranked_hits[:limit]
