import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.ingest.curation import CuratedIngestionValidator
from backend.core.retrieval.vectors.faiss_index import FaissIndex
from backend.core.schemas.curation import CuratedDocument, IngestionManifestRecord

INDEXES_DIR = "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod/backend/core/retrieval/vectors"
TENANT_ID = "demo"


def main(manifest_path: str):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(path)

    faiss_cache = {}
    indexed = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = IngestionManifestRecord(**payload)
            ok, reason = CuratedIngestionValidator.validate_manifest_record(record)
            if not ok:
                print(f"Reject line {line_number}: {reason}")
                continue

            if not record.content:
                print(f"Reject line {line_number}: content is required for indexing")
                continue

            document = CuratedIngestionValidator.build_document(
                source_path=record.source_path,
                source_uri=record.metadata.get("source_uri", record.source_path),
                title=record.title,
                module=record.module,
                task_type=record.task_type,
                doc_type=record.doc_type,
                trusted_schema_objects=record.trusted_schema_objects,
                quality_score=record.quality_score,
                source_system=record.source_system,
                content=record.content,
                metadata=record.metadata,
            )
            chunk = CuratedIngestionValidator.build_chunk(document, document.content, 0)
            chunk_payload = CuratedIngestionValidator.chunk_payload(chunk)
            corpus = chunk_payload["metadata"]["corpus"]

            if corpus not in faiss_cache:
                faiss_cache[corpus] = FaissIndex(tenant_id=TENANT_ID, indexes_dir=INDEXES_DIR, corpus=corpus)
            faiss_cache[corpus].add_chunks_list([chunk_payload], batch_size=1)
            indexed += 1

    print(f"Indexed {indexed} curated manifest rows from {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ingest_manifest_jsonl.py /path/to/manifest.jsonl")
    main(sys.argv[1])
