import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.schemas.curation import (
    AuthorityTier,
    CorpusType,
    CuratedChunk,
    CuratedDocument,
    DocType,
    IngestionManifestRecord,
    SourceSystem,
)
from core.schemas.router import ModuleFamily, module_families_for_value


REQUIRED_CHUNK_FIELDS = {
    CorpusType.SCHEMA: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.SQL: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.DOCS: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.TROUBLESHOOTING: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.SQL_EXAMPLES: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.SCHEMA_METADATA: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "trusted_schema_objects",
        "quality_score",
        "content_hash",
        "source_system",
    },
    CorpusType.FAST_FORMULA: {
        "source_path",
        "title",
        "module",
        "task_type",
        "doc_type",
        "quality_score",
        "content_hash",
        "source_system",
    },
}

SQL_REJECT_PATTERNS = [
    r"(?i)\bselect\s+\*\b",
    r"(?i)\bfrom\s+dual\b",
    r"(?i)^\s*select\s+'[^']+'\s*$",
    r"(?i)\btodo\b",
    r"(?i)\bplaceholder\b",
]


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def stable_hash(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update((part or "").encode("utf-8"))
    return hasher.hexdigest()


def infer_module_family(module: str) -> str:
    families = module_families_for_value(module)
    if families:
        return next(iter(families))
    return ModuleFamily.UNKNOWN.value


def infer_corpus(doc_type: DocType) -> CorpusType:
    if doc_type in {DocType.SCHEMA_TABLE, DocType.SCHEMA_VIEW, DocType.SCHEMA_RELATION}:
        return CorpusType.SCHEMA
    if doc_type == DocType.VERIFIED_SQL:
        return CorpusType.SQL
    if doc_type == DocType.SQL_EXAMPLE:
        return CorpusType.SQL_EXAMPLES
    if doc_type == DocType.SCHEMA_METADATA:
        return CorpusType.SCHEMA_METADATA
    if doc_type in {
        DocType.FUNCTIONAL_DOC,
        DocType.PROCEDURE_DOC,
        DocType.NAVIGATION_DOC,
        DocType.SETUP_DOC,
    }:
        return CorpusType.DOCS
    if doc_type == DocType.TROUBLESHOOTING_DOC:
        return CorpusType.TROUBLESHOOTING
    if doc_type in {
        DocType.FAST_FORMULA_EXAMPLE,
        DocType.FAST_FORMULA_DOC,
    }:
        return CorpusType.FAST_FORMULA
    return CorpusType.REJECT


class CuratedIngestionValidator:
    """
    Source-agnostic validation and normalization for curated SLM corpora.
    """

    @staticmethod
    def required_fields(corpus: CorpusType) -> set[str]:
        return REQUIRED_CHUNK_FIELDS.get(corpus, set())

    @staticmethod
    def validate_manifest_record(record: IngestionManifestRecord) -> Tuple[bool, Optional[str]]:
        corpus = record.corpus or infer_corpus(record.doc_type)
        if corpus == CorpusType.REJECT:
            return False, "Document type maps to reject_corpus."

        if record.quality_score < 0.65:
            return False, "quality_score below curated threshold."

        if not record.trusted_schema_objects and corpus in {
            CorpusType.SCHEMA,
            CorpusType.SQL,
            CorpusType.SQL_EXAMPLES,
            CorpusType.SCHEMA_METADATA,
        }:
            return False, "trusted_schema_objects required for schema/sql corpora."

        required = CuratedIngestionValidator.required_fields(corpus)
        payload = record.model_dump()
        missing = []
        for field in required:
            value = payload.get(field)
            if field == "trusted_schema_objects":
                if value is None:
                    missing.append(field)
                continue
            if value in (None, "", [], {}, ()):
                missing.append(field)
        if missing:
            return False, f"missing required metadata: {', '.join(sorted(missing))}"

        if record.source_system in {SourceSystem.ORACLE_DOCS, SourceSystem.ORACLEWINGS_REPO}:
            if not (record.source_uri or record.metadata.get("source_uri")):
                return False, "source_uri required for oracle docs and oraclewings repo content."

        return True, None

    @staticmethod
    def reject_sql(sql: str) -> Optional[str]:
        normalized = sql.strip()
        if not normalized:
            return "empty sql"
        for pattern in SQL_REJECT_PATTERNS:
            if re.search(pattern, normalized):
                return f"rejected by sql rule: {pattern}"
        return None

    @staticmethod
    def build_document(
        *,
        source_path: str,
        title: str,
        module: str,
        task_type: str,
        doc_type: DocType,
        trusted_schema_objects: List[str],
        quality_score: float,
        source_system: SourceSystem,
        content: str,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> CuratedDocument:
        corpus = infer_corpus(doc_type)
        normalized_content = _normalized_text(content)
        content_hash = stable_hash(corpus.value, normalized_content)
        metadata = dict(metadata or {})
        metadata.setdefault("source_path", source_path)
        metadata.setdefault("source_uri", source_uri or source_path)
        metadata.setdefault("canonical_uri", metadata.get("source_uri", source_uri or source_path))
        metadata.setdefault("title", title)
        metadata.setdefault("module", module)
        metadata.setdefault("module_family", infer_module_family(module))
        metadata.setdefault("task_type", task_type)
        metadata.setdefault("doc_type", doc_type.value)
        metadata.setdefault("trusted_schema_objects", trusted_schema_objects)
        metadata.setdefault("quality_score", quality_score)
        metadata.setdefault("source_system", source_system.value)
        metadata.setdefault(
            "authority_tier",
            AuthorityTier.OFFICIAL.value if source_system == SourceSystem.ORACLE_DOCS else AuthorityTier.SECONDARY.value,
        )
        metadata.setdefault("corpus", corpus.value)
        metadata.setdefault("content_hash", content_hash)

        doc = CuratedDocument(
            document_id=document_id or stable_hash(source_path, title)[:24],
            source_path=source_path,
            source_uri=source_uri or source_path,
            title=title,
            module=module,
            task_type=task_type,
            doc_type=doc_type,
            trusted_schema_objects=trusted_schema_objects,
            quality_score=quality_score,
            source_system=source_system,
            corpus=corpus,
            content=content,
            content_hash=content_hash,
            metadata=metadata,
        )
        manifest = IngestionManifestRecord(
            source_path=doc.source_path,
            source_uri=doc.source_uri,
            title=doc.title,
            module=doc.module,
            task_type=doc.task_type,
            doc_type=doc.doc_type,
            trusted_schema_objects=doc.trusted_schema_objects,
            quality_score=doc.quality_score,
            content_hash=doc.content_hash,
            source_system=doc.source_system,
            corpus=doc.corpus,
            content=doc.content,
            canonical_uri=metadata.get("canonical_uri"),
            doc_release=str(metadata.get("doc_release") or "") or None,
            authority_tier=AuthorityTier(str(metadata.get("authority_tier")))
            if metadata.get("authority_tier") in {tier.value for tier in AuthorityTier}
            else None,
            metadata=doc.metadata,
        )
        ok, reason = CuratedIngestionValidator.validate_manifest_record(manifest)
        if not ok:
            raise ValueError(reason)
        return doc

    @staticmethod
    def build_chunk(
        document: CuratedDocument,
        content: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CuratedChunk:
        normalized_content = _normalized_text(content)
        content_hash = stable_hash(document.corpus.value, normalized_content)
        chunk_id = f"{document.document_id}:{chunk_index}:{content_hash[:12]}"
        chunk_metadata = dict(document.metadata)
        chunk_metadata.update(metadata or {})
        chunk_metadata.setdefault("chunk_id", chunk_id)
        chunk_metadata.setdefault("source_uri", document.source_uri)
        chunk_metadata.setdefault("canonical_uri", document.metadata.get("canonical_uri", document.source_uri))

        chunk = CuratedChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            content=content,
            source_path=document.source_path,
            title=document.title,
            module=document.module,
            task_type=document.task_type,
            doc_type=document.doc_type,
            trusted_schema_objects=document.trusted_schema_objects,
            quality_score=document.quality_score,
            content_hash=content_hash,
            source_system=document.source_system,
            corpus=document.corpus,
            metadata=chunk_metadata,
        )

        manifest = IngestionManifestRecord(
            source_path=chunk.source_path,
            source_uri=chunk.metadata.get("source_uri"),
            title=chunk.title,
            module=chunk.module,
            task_type=chunk.task_type,
            doc_type=chunk.doc_type,
            trusted_schema_objects=chunk.trusted_schema_objects,
            quality_score=chunk.quality_score,
            chunk_id=chunk.chunk_id,
            content_hash=chunk.content_hash,
            source_system=chunk.source_system,
            corpus=chunk.corpus,
            content=chunk.content,
            canonical_uri=chunk.metadata.get("canonical_uri"),
            doc_release=str(chunk.metadata.get("doc_release") or "") or None,
            authority_tier=AuthorityTier(str(chunk.metadata.get("authority_tier")))
            if chunk.metadata.get("authority_tier") in {tier.value for tier in AuthorityTier}
            else None,
            metadata=chunk.metadata,
        )
        ok, reason = CuratedIngestionValidator.validate_manifest_record(manifest)
        if not ok:
            raise ValueError(reason)
        return chunk

    @staticmethod
    def chunk_payload(chunk: CuratedChunk) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "metadata": {
                **chunk.metadata,
                "source_path": chunk.source_path,
                "canonical_uri": chunk.metadata.get("canonical_uri", chunk.metadata.get("source_uri", chunk.source_path)),
                "filename": chunk.title,
                "title": chunk.title,
                "module": chunk.module,
                "module_family": chunk.metadata.get("module_family", infer_module_family(chunk.module)),
                "task_type": chunk.task_type,
                "doc_type": chunk.doc_type.value,
                "trusted_schema_objects": chunk.trusted_schema_objects,
                "quality_score": chunk.quality_score,
                "source_system": chunk.source_system.value,
                "authority_tier": chunk.metadata.get("authority_tier"),
                "source_uri": chunk.metadata.get("source_uri", chunk.source_path),
                "corpus": chunk.corpus.value,
                "content_hash": chunk.content_hash,
            },
        }

    @staticmethod
    def is_curated_metadata(metadata: Dict[str, Any], corpus: Optional[str] = None) -> bool:
        metadata = metadata or {}
        resolved_corpus = corpus or metadata.get("corpus")
        if not resolved_corpus:
            return False
        try:
            corpus_type = CorpusType(resolved_corpus)
        except ValueError:
            return False

        required = CuratedIngestionValidator.required_fields(corpus_type)
        for field in required:
            value = metadata.get(field)
            if field == "trusted_schema_objects":
                if corpus_type in {
                    CorpusType.SCHEMA,
                    CorpusType.SQL,
                    CorpusType.SQL_EXAMPLES,
                    CorpusType.SCHEMA_METADATA,
                } and value in (None, "", [], {}, ()):
                    return False
                continue
            if value in (None, "", [], {}, ()):
                return False
        source_system = str(metadata.get("source_system") or "")
        if source_system in {SourceSystem.ORACLE_DOCS.value, SourceSystem.ORACLEWINGS_REPO.value}:
            if metadata.get("source_uri") in (None, "", [], {}, ()):
                return False
        return True
