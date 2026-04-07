import asyncio
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.scripts.index_golden_sql import main as index_sql_corpus
from backend.scripts.index_schema_corpus import main as index_schema_corpus
from backend.scripts.reindex_metadata import main as rebuild_metadata_registry
import backend.core.rag.engine as rag_engine_module
from backend.core.grounding.trusted_registry import get_default_registry
from backend.core.rag.engine import FAIL_CLOSED_MESSAGE, RAGEngine
from backend.core.schemas.api import ChatRequest, Message, Role
from backend.core.schemas.router import module_families_for_value

INDEX_ROOT = BASE_DIR / "backend" / "core" / "retrieval" / "vectors" / "faiss"
TENANT_ID = "demo"
VALIDATION_SOURCE = ROOT_DIR / "test_cases" / "oracle_fusion_llm_test_cases_1000.json"
VALIDATION_OUTPUT = BASE_DIR / "curated_activation_validation.jsonl"
SUMMARY_OUTPUT = BASE_DIR / "curated_activation_summary.json"


def load_bootstrap_docs():
    script_path = BASE_DIR / "scripts" / "bootstrap_rag.py"
    spec = importlib.util.spec_from_file_location("bootstrap_rag", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bootstrap


def archive_legacy_flat_store() -> str | None:
    legacy_files = [
        INDEX_ROOT / "faiss.index",
        INDEX_ROOT / "metadata.sqlite",
    ]
    existing = [path for path in legacy_files if path.exists()]
    if not existing:
        return None

    archive_dir = INDEX_ROOT / f"_legacy_flat_store_disabled_{time.strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), archive_dir / path.name)
    return str(archive_dir)


def corpus_paths(corpus: str) -> Dict[str, Path]:
    base = INDEX_ROOT / TENANT_ID / corpus
    return {
        "dir": base,
        "db": base / "metadata.sqlite",
        "index": base / "faiss.index",
    }


def corpus_stats(corpus: str) -> Dict[str, Any]:
    paths = corpus_paths(corpus)
    stats: Dict[str, Any] = {
        "corpus": corpus,
        "dir_exists": paths["dir"].exists(),
        "index_exists": paths["index"].exists(),
        "db_exists": paths["db"].exists(),
        "chunks": 0,
        "unique_content_hashes": 0,
        "samples": [],
    }
    if not paths["db"].exists():
        return stats

    with sqlite3.connect(paths["db"]) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS chunks, COUNT(DISTINCT content_hash) AS unique_hashes FROM chunks"
        ).fetchone()
        stats["chunks"] = int(row[0]) if row else 0
        stats["unique_content_hashes"] = int(row[1]) if row else 0
        sample_rows = conn.execute(
            "SELECT chunk_id, document_id, metadata, substr(content, 1, 220) FROM chunks ORDER BY vector_id LIMIT 3"
        ).fetchall()
    for chunk_id, document_id, metadata_json, content_preview in sample_rows:
        metadata = json.loads(metadata_json)
        stats["samples"].append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": metadata.get("title"),
                "module": metadata.get("module"),
                "task_type": metadata.get("task_type"),
                "doc_type": metadata.get("doc_type"),
                "source_path": metadata.get("source_path"),
                "corpus": metadata.get("corpus"),
                "content_preview": content_preview,
            }
        )
    return stats


def load_validation_cases(sample_size: int) -> List[Dict[str, Any]]:
    payload = json.loads(VALIDATION_SOURCE.read_text(encoding="utf-8"))
    cases = payload["test_cases"]
    step = max(1, len(cases) // sample_size)
    selected = [cases[index] for index in range(0, len(cases), step)]
    return selected[:sample_size]


def extract_sql(output: str) -> str | None:
    if not output or output == FAIL_CLOSED_MESSAGE:
        return None

    fenced = re.search(r"```sql\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    inline = re.search(r"(?is)\bselect\b.+?(?:;|$)", output)
    if inline:
        return inline.group(0).strip()
    return None


def classify_failure(record: Dict[str, Any]) -> str:
    if record["verifier_passed"]:
        return "passed"

    if record.get("runtime_error"):
        return "runtime_error"

    if record["detected_module"] in {"Unknown", "UNKNOWN"}:
        return "wrong module routing"

    corpora = set(record["retrieved_corpora"])
    task_type = record["intent_detected"]

    if task_type in {"table_lookup", "sql_generation", "report_logic", "troubleshooting"} and "schema_corpus" not in corpora:
        return "missing schema"
    if task_type in {"sql_generation", "report_logic"} and "sql_corpus" not in corpora:
        return "missing SQL example"
    if task_type in {"procedure", "navigation", "troubleshooting", "general", "integration"} and "docs_corpus" not in corpora:
        return "insufficient docs"
    return "over-rejection (too strict)"


def family_for(value: Any) -> str:
    families = module_families_for_value(str(value)) if value else {"UNKNOWN"}
    return next(iter(families)) if families else "UNKNOWN"


async def run_validation(sample_size: int = 50, max_tokens: int = 80) -> Dict[str, Any]:
    cases = load_validation_cases(sample_size)
    original_max_retries = rag_engine_module.MAX_VERIFICATION_RETRIES
    rag_engine_module.MAX_VERIFICATION_RETRIES = 0
    engine = RAGEngine()
    engine.reranker.rerank = lambda query, documents, top_k=8: documents[:top_k]
    tenant = SimpleNamespace(id=TENANT_ID)
    registry = get_default_registry()
    results: List[Dict[str, Any]] = []

    try:
        for index, case in enumerate(cases, start=1):
            question = case["question"]
            if index == 1 or index % 5 == 0 or index == len(cases):
                print(f"[validation] case {index}/{len(cases)}: {case['id']}")
            request = ChatRequest(
                messages=[Message(role=Role.USER, content=question)],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=0.9,
                repeat_penalty=1.1,
            )

            try:
                response = await engine.chat(db=None, tenant=tenant, request=request, require_citations=True)
                output = response.choices[0]["message"]["content"]
                retrieved_chunks = response.retrieved_chunks or []
                schema_objects = sorted(
                    {
                        obj
                        for chunk in retrieved_chunks
                        for obj in ((chunk.get("metadata") or {}).get("trusted_schema_objects") or [])
                    }
                )
                record: Dict[str, Any] = {
                    "id": case["id"],
                    "question": question,
                    "benchmark_module": case.get("module"),
                    "category": case.get("category"),
                    "module_detected": response.audit.get("module"),
                    "module_family_detected": response.audit.get("module_family"),
                    "intent_detected": response.audit.get("task_type"),
                    "detected_module": response.audit.get("module"),
                    "retrieved_chunks": [
                        {
                            "citation_id": chunk.get("citation_id"),
                            "title": chunk.get("title"),
                            "module": (chunk.get("metadata") or {}).get("module"),
                            "corpus": (chunk.get("metadata") or {}).get("corpus"),
                            "score": float(chunk.get("combined_score") or chunk.get("score") or 0.0),
                            "source_uri": chunk.get("source_uri"),
                        }
                        for chunk in retrieved_chunks
                    ],
                    "retrieved_corpora": [
                        (chunk.get("metadata") or {}).get("corpus")
                        for chunk in retrieved_chunks
                        if (chunk.get("metadata") or {}).get("corpus")
                    ],
                    "schema_objects_used": schema_objects,
                    "unknown_schema_usage": any(
                        (registry.get_entry(obj) or {}).get("owning_module_family") == "UNKNOWN"
                        for obj in schema_objects
                    ),
                    "sql_index_used": any(
                        (chunk.get("metadata") or {}).get("corpus") == "sql_corpus"
                        for chunk in retrieved_chunks
                    ),
                    "sql_generated": extract_sql(output),
                    "verifier_status": response.audit.get("verification_status"),
                    "verifier_passed": response.audit.get("verification_status") == "PASSED",
                    "rejected": output == FAIL_CLOSED_MESSAGE,
                    "output": output,
                    "runtime_error": None,
                }
            except Exception as exc:
                record = {
                    "id": case["id"],
                    "question": question,
                    "benchmark_module": case.get("module"),
                    "category": case.get("category"),
                    "module_detected": None,
                    "module_family_detected": "UNKNOWN",
                    "intent_detected": None,
                    "detected_module": "UNKNOWN",
                    "retrieved_chunks": [],
                    "retrieved_corpora": [],
                    "schema_objects_used": [],
                    "unknown_schema_usage": False,
                    "sql_index_used": False,
                    "sql_generated": None,
                    "verifier_status": "RUNTIME_ERROR",
                    "verifier_passed": False,
                    "rejected": True,
                    "output": FAIL_CLOSED_MESSAGE,
                    "runtime_error": str(exc),
                }

            record["failure_category"] = classify_failure(record)
            results.append(record)
    finally:
        rag_engine_module.MAX_VERIFICATION_RETRIES = original_max_retries

    with open(VALIDATION_OUTPUT, "w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    failure_counts: Dict[str, int] = {}
    for record in results:
        failure_counts[record["failure_category"]] = failure_counts.get(record["failure_category"], 0) + 1

    relevant_sql_records = [
        record for record in results if record.get("intent_detected") in {"sql_generation", "report_logic", "troubleshooting"}
    ]

    summary = {
        "sample_size": len(results),
        "validation_mode": {
            "max_tokens": max_tokens,
            "max_verification_retries": 0,
        },
        "passes": sum(1 for record in results if record["verifier_passed"]),
        "rejections": sum(1 for record in results if record["rejected"]),
        "rejection_rate": round(sum(1 for record in results if record["rejected"]) / max(len(results), 1) * 100, 2),
        "routing_family_accuracy": round(
            sum(1 for record in results if family_for(record.get("benchmark_module")) == family_for(record.get("module_family_detected") or record.get("module_detected")))
            / max(len(results), 1)
            * 100,
            2,
        ),
        "sql_index_usage_pct": round(
            sum(1 for record in results if record.get("sql_index_used")) / max(len(results), 1) * 100,
            2,
        ),
        "sql_index_usage_relevant_pct": round(
            sum(1 for record in relevant_sql_records if record.get("sql_index_used")) / max(len(relevant_sql_records), 1) * 100,
            2,
        ),
        "unknown_usage_pct": round(
            sum(1 for record in results if record.get("unknown_schema_usage")) / max(len(results), 1) * 100,
            2,
        ),
        "unknown_route_rate": round(
            sum(1 for record in results if family_for(record.get("module_family_detected") or record.get("module_detected")) == "UNKNOWN")
            / max(len(results), 1)
            * 100,
            2,
        ),
        "failures": failure_counts,
        "good_examples": [record for record in results if record["verifier_passed"]][:3],
        "rejected_examples": [record for record in results if record["rejected"]][:3],
        "retrieval_samples": results[:5],
        "validation_output": str(VALIDATION_OUTPUT),
    }

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    return summary


def activate(sample_size: int = 50) -> Dict[str, Any]:
    bootstrap_docs = load_bootstrap_docs()
    archived_legacy = archive_legacy_flat_store()

    print("[activation] rebuilding trusted metadata registry")
    rebuild_metadata_registry()
    print("[activation] rebuilding schema_corpus")
    index_schema_corpus()
    print("[activation] rebuilding sql_corpus")
    index_sql_corpus()
    print("[activation] rebuilding docs_corpus")
    bootstrap_docs()
    print("[activation] refreshing trusted metadata registry with docs signals")
    rebuild_metadata_registry()
    print("[activation] refreshing schema_corpus with docs-informed families")
    index_schema_corpus()

    index_summary = {
        "legacy_flat_store_archived_to": archived_legacy,
        "schema_index": corpus_stats("schema_corpus"),
        "sql_index": corpus_stats("sql_corpus"),
        "docs_index": corpus_stats("docs_corpus"),
    }
    validation_summary = asyncio.run(run_validation(sample_size=sample_size))

    final_summary = {
        "index_summary": index_summary,
        "validation_summary": validation_summary,
        "summary_output": str(SUMMARY_OUTPUT),
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=True))
    return final_summary


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    activate(sample_size=sample_size)
