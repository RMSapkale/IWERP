import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import get_default_registry
from backend.core.ingest.curation import CuratedIngestionValidator
from backend.core.retrieval.vectors.faiss_index import FaissIndex
from backend.core.schemas.curation import DocType, SourceSystem

INDEXES_DIR = "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod/backend/core/retrieval/vectors"
TENANT_ID = "demo"


def main():
    registry = get_default_registry()
    faiss_index = FaissIndex(tenant_id=TENANT_ID, indexes_dir=INDEXES_DIR, corpus="schema_corpus")
    faiss_index.reset()
    chunks = []

    for object_name in sorted(registry.objects):
        schema_chunk = registry.build_schema_chunk(object_name)
        if not schema_chunk:
            continue

        entry = registry.get_entry(object_name)
        doc_type = DocType.SCHEMA_VIEW if entry and entry.get("object_type") == "view" else DocType.SCHEMA_TABLE
        document = CuratedIngestionValidator.build_document(
            source_path=schema_chunk["metadata"]["source_uri"],
            source_uri=schema_chunk["metadata"]["source_uri"],
            title=object_name,
            module=schema_chunk["metadata"]["module"],
            task_type="table_lookup",
            doc_type=doc_type,
            trusted_schema_objects=[object_name],
            quality_score=float(schema_chunk["metadata"]["quality_score"]),
            source_system=SourceSystem.METADATA,
            content=schema_chunk["content"],
            document_id=f"SCHEMA::{object_name}",
        )
        chunk = CuratedIngestionValidator.build_chunk(document, schema_chunk["content"], 0)
        chunks.append(CuratedIngestionValidator.chunk_payload(chunk))

    print(f"Indexing {len(chunks)} curated schema chunks...")
    faiss_index.add_chunks_list(chunks, batch_size=64)
    print(f"Schema corpus indexing complete. Stats: {faiss_index.stats()}")


if __name__ == "__main__":
    main()
