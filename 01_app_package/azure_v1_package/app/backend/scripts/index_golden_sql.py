import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import get_default_registry
from backend.core.grounding.verifier import Verifier
from backend.core.ingest.curation import CuratedIngestionValidator
from backend.core.retrieval.vectors.faiss_index import FaissIndex
from backend.core.schemas.curation import DocType, SourceSystem

ORG_SQL_DIR = "/Users/integrationwings/Desktop/LLM_Wrap/oracle-fusion-slm/oracle-fusion-slm/data/source_raw/organized_sql"
INDEXES_DIR = "/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod/backend/core/retrieval/vectors"
TENANT_ID = "demo"
PATH_MODULE_HINTS = [
    (r"(?i)(accounts payable|\bAP\b|payables)", "Payables"),
    (r"(?i)(accounts receivable|\bAR\b|receivables)", "Receivables"),
    (r"(?i)(general ledger|\bGL\b)", "General Ledger"),
    (r"(?i)(cash management|bank|reconciliation|\bCE\b)", "Cash Management"),
    (r"(?i)(budgetary control|subledger|financials|payments)", "Financials"),
    (r"(?i)(procurement|purchasing|supplier|requisition|\bPO\b|\bPOZ\b|\bPOR\b)", "Procurement"),
    (r"(?i)(inventory|shipping|order management|planning|manufacturing|supply chain|\bINV\b|\bDOO\b|\bMSC\b)", "SCM"),
    (r"(?i)(projects|ppm|grant)", "Projects"),
    (r"(?i)(hcm|payroll|employee|absence|recruiting|benefits)", "HCM"),
]


def parse_xdm_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    queries = re.split(r"\[Query \d+\]", content)
    return [query.strip() for query in queries[1:] if query.strip() and len(query.strip()) > 20]


def parse_json_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        try:
            data = json.load(handle)
        except Exception:
            return []
    if isinstance(data, dict):
        sql = data.get("sql") or data.get("query") or data.get("content")
        return [str(sql)] if sql else []
    return []


def extract_tables(sql, registry):
    candidates = set(re.findall(r"(?i)(?:from|join)\s+([A-Z][A-Z0-9_$.]+)", sql))
    resolved = []
    for candidate in candidates:
        canonical = registry.resolve_object_name(candidate)
        if canonical:
            resolved.append(canonical)
    return sorted(set(resolved))


def infer_module(tables, registry):
    for table_name in tables:
        entry = registry.get_entry(table_name)
        if entry and entry.get("owning_module") not in {"UNKNOWN", "Common"}:
            return entry["owning_module"]
    return "Common"


def infer_module_from_path(rel_path):
    for pattern, module in PATH_MODULE_HINTS:
        if re.search(pattern, rel_path):
            return module
    return None


def infer_task_type(rel_path: str) -> str:
    if re.search(r"(?i)(exception|exceptions|error|errors|failed|failure|recon|reconciliation|validation)", rel_path):
        return "troubleshooting"
    if re.search(r"(?i)(report|analysis|summary|extract|dashboard)", rel_path):
        return "report_logic"
    return "sql_generation"


def main():
    print(f"Starting curated SQL ingestion for tenant: {TENANT_ID}")
    registry = get_default_registry()
    verifier = Verifier()
    faiss_index = FaissIndex(tenant_id=TENANT_ID, indexes_dir=INDEXES_DIR, corpus="sql_corpus")
    faiss_index.reset()

    chunks = []
    processed_files = 0
    rejected_sql = 0
    failed_verifier = 0
    missing_objects = 0

    for root, _, files in os.walk(ORG_SQL_DIR):
        for file_name in files:
            filepath = os.path.join(root, file_name)
            rel_path = os.path.relpath(filepath, ORG_SQL_DIR)
            file_queries = []

            if file_name.endswith(".xdm.txt"):
                file_queries = parse_xdm_file(filepath)
            elif file_name.endswith(".json"):
                file_queries = parse_json_file(filepath)

            if not file_queries:
                continue

            processed_files += 1
            for index, sql in enumerate(file_queries):
                if CuratedIngestionValidator.reject_sql(sql):
                    rejected_sql += 1
                    continue
                valid, reason = verifier.verify_sql(sql)
                if not valid:
                    failed_verifier += 1
                    continue

                trusted_objects = extract_tables(sql, registry)
                if not trusted_objects:
                    missing_objects += 1
                    continue

                module = infer_module(trusted_objects, registry)
                if module in {"UNKNOWN", "Common"}:
                    module = infer_module_from_path(rel_path) or module
                task_type = infer_task_type(rel_path)
                document = CuratedIngestionValidator.build_document(
                    source_path=filepath,
                    source_uri=rel_path,
                    title=rel_path,
                    module=module,
                    task_type=task_type,
                    doc_type=DocType.VERIFIED_SQL,
                    trusted_schema_objects=trusted_objects,
                    quality_score=0.95,
                    source_system=SourceSystem.REPO,
                    content=sql,
                    document_id=f"SQL::{rel_path}::{index}",
                )
                chunk = CuratedIngestionValidator.build_chunk(document, sql, 0)
                chunks.append(CuratedIngestionValidator.chunk_payload(chunk))

    print(
        f"Prepared {len(chunks)} curated SQL chunks from {processed_files} files. "
        f"Rejected_sql={rejected_sql} verifier_failures={failed_verifier} missing_objects={missing_objects}"
    )
    if chunks:
        faiss_index.add_chunks_list(chunks, batch_size=32)
        print(f"Curated SQL ingestion complete. Stats: {faiss_index.stats()}")
    else:
        print("No curated SQL patterns passed validation.")


if __name__ == "__main__":
    main()
