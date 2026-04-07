import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.rag.engine import FAIL_CLOSED_MESSAGE, RAGEngine
from backend.core.schemas.api import ChatRequest, Message, Role
from backend.core.schemas.router import module_families_for_value

VALIDATION_SOURCE = ROOT_DIR / "test_cases" / "oracle_fusion_llm_test_cases_1000.json"
OUTPUT_DIR = BASE_DIR / "real_model_validation"
TARGET_BUCKETS = [
    "Payables",
    "Receivables",
    "General Ledger",
    "Procurement",
    "SCM",
    "HCM",
]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DOCS_EXPECTED_TASKS = {"procedure", "navigation", "troubleshooting", "general", "integration", "summary"}
SQL_EXPECTED_TASKS = {"sql_generation", "report_logic"}
GENERIC_PATTERNS = [
    "relevant work area",
    "complete the required attributes",
    "submit or save it according to your approval",
    "based on the provided documentation",
]
INFERENCE_PATTERNS = [
    "not explicitly mentioned",
    "however, based on",
    "it can be inferred",
    "it can be assumed",
]


def family_for(value: Any) -> str:
    families = module_families_for_value(str(value)) if value else {"UNKNOWN"}
    return next(iter(families)) if families else "UNKNOWN"


def case_bucket(case: Dict[str, Any]) -> str | None:
    module = str(case.get("module") or "")
    family = family_for(module)
    if module in {"Payables", "Receivables", "General Ledger"}:
        return module
    if module == "Procurement" or family == "Procurement":
        return "Procurement"
    if module == "HCM" or family == "HCM":
        return "HCM"
    if module in {
        "SCM",
        "Supply Chain",
        "Inventory Management",
        "Order Management",
        "Manufacturing",
        "Shipping",
        "Planning",
        "Product Management",
    } or family == "SCM":
        return "SCM"
    return None


def balanced_case_selection(sample_size: int) -> List[Dict[str, Any]]:
    payload = json.loads(VALIDATION_SOURCE.read_text(encoding="utf-8"))
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case in payload["test_cases"]:
        bucket = case_bucket(case)
        if bucket:
            grouped[bucket][str(case.get("difficulty", "medium")).lower()].append(case)

    per_bucket = max(1, sample_size // len(TARGET_BUCKETS))
    selected: List[Dict[str, Any]] = []
    for bucket in TARGET_BUCKETS:
        bucket_cases = []
        by_difficulty = grouped.get(bucket, {})
        positions = {difficulty: 0 for difficulty in DIFFICULTY_ORDER}
        while len(bucket_cases) < per_bucket:
            progressed = False
            for difficulty in DIFFICULTY_ORDER:
                cases = by_difficulty.get(difficulty, [])
                index = positions[difficulty]
                if index < len(cases):
                    bucket_cases.append(cases[index])
                    positions[difficulty] += 1
                    progressed = True
                if len(bucket_cases) >= per_bucket:
                    break
            if not progressed:
                break
        selected.extend(bucket_cases[:per_bucket])

    return selected[: sample_size - (sample_size % len(TARGET_BUCKETS))] or selected[:sample_size]


def parse_bucket_plan(bucket_plan: Optional[str]) -> Dict[str, int]:
    if not bucket_plan:
        return {}
    parsed: Dict[str, int] = {}
    for raw_part in bucket_plan.split(","):
        part = raw_part.strip()
        if not part or ":" not in part:
            continue
        bucket, count = part.rsplit(":", 1)
        try:
            parsed[bucket.strip()] = max(0, int(count))
        except ValueError:
            continue
    return parsed


def focused_case_selection(bucket_plan: Dict[str, int]) -> List[Dict[str, Any]]:
    payload = json.loads(VALIDATION_SOURCE.read_text(encoding="utf-8"))
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case in payload["test_cases"]:
        bucket = case_bucket(case)
        if bucket in bucket_plan:
            grouped[bucket][str(case.get("difficulty", "medium")).lower()].append(case)

    selected: List[Dict[str, Any]] = []
    for bucket, requested_count in bucket_plan.items():
        by_difficulty = grouped.get(bucket, {})
        positions = {difficulty: 0 for difficulty in DIFFICULTY_ORDER}
        bucket_cases: List[Dict[str, Any]] = []
        while len(bucket_cases) < requested_count:
            progressed = False
            for difficulty in DIFFICULTY_ORDER:
                cases = by_difficulty.get(difficulty, [])
                index = positions[difficulty]
                if index < len(cases):
                    bucket_cases.append(cases[index])
                    positions[difficulty] += 1
                    progressed = True
                if len(bucket_cases) >= requested_count:
                    break
            if not progressed:
                break
        selected.extend(bucket_cases[:requested_count])
    return selected


def extract_sql(output: str) -> str | None:
    if not output or output == FAIL_CLOSED_MESSAGE:
        return None

    section = re.search(r"\[SQL\]\s*(.*?)(?:\n\[[A-Za-z]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    if section:
        sql_block = section.group(1).strip()
        return sql_block or None

    fenced = re.search(r"```sql\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return None


def is_safe_rejection(output: str) -> bool:
    normalized = (output or "").strip()
    return FAIL_CLOSED_MESSAGE in normalized


def quality_tag(record: Dict[str, Any]) -> str:
    output = (record.get("output") or "").lower()
    intent = record.get("intent_detected") or ""
    benchmark_bucket = record.get("benchmark_bucket")
    detected_module = record.get("module_detected")
    detected_family = record.get("module_family_detected")

    if record.get("runtime_error"):
        return "runtime_error"
    if record.get("rejected"):
        return "over_rejection"
    if benchmark_bucket in {"Payables", "Receivables", "General Ledger"} and detected_module != benchmark_bucket:
        return "wrong_module_grounded"
    if benchmark_bucket in {"Procurement", "SCM", "HCM"} and family_for(detected_family or detected_module) != benchmark_bucket:
        return "wrong_module_grounded"
    if record.get("sql_generated") and intent not in SQL_EXPECTED_TASKS and intent != "troubleshooting":
        return "sql_misuse"
    if FAIL_CLOSED_MESSAGE.lower() in output and output.strip() != FAIL_CLOSED_MESSAGE.lower():
        return "mixed_refusal"
    if intent in DOCS_EXPECTED_TASKS and not record.get("docs_passed_to_prompt_count"):
        return "ranking_issue_missing_docs"
    if any(pattern in output for pattern in INFERENCE_PATTERNS):
        return "weak_inference"
    if any(pattern in output for pattern in GENERIC_PATTERNS):
        return "weak_generic"
    if len((record.get("output") or "").split()) < 35 and intent in DOCS_EXPECTED_TASKS:
        return "weak_explanation"
    return "likely_correct"


def exact_module_top_hit(record: Dict[str, Any]) -> bool:
    docs = record.get("retrieved_docs") or []
    if not docs:
        return False
    top = docs[0]
    benchmark_bucket = record.get("benchmark_bucket")
    top_module = top.get("module")
    top_family = top.get("module_family")
    if benchmark_bucket in {"Payables", "Receivables", "General Ledger"}:
        return top_module == benchmark_bucket
    return family_for(top_module or top_family) == benchmark_bucket


def same_family_bleed(record: Dict[str, Any]) -> bool:
    benchmark_bucket = record.get("benchmark_bucket")
    if benchmark_bucket not in {"Payables", "Receivables", "General Ledger"}:
        return False
    expected_family = family_for(benchmark_bucket)
    for doc in record.get("retrieved_docs") or []:
        doc_module = doc.get("module")
        doc_family = family_for(doc_module or doc.get("module_family"))
        if doc_family == expected_family and doc_module not in {benchmark_bucket, None, ""}:
            return True
    return False


def summarize_records(records: List[Dict[str, Any]], label: str, max_tokens: int) -> Dict[str, Any]:
    quality_counts = Counter(record["quality_tag"] for record in records)
    sql_records = [record for record in records if record.get("intent_detected") in SQL_EXPECTED_TASKS or record.get("sql_generated")]
    docs_records = [record for record in records if record.get("intent_detected") in DOCS_EXPECTED_TASKS]
    rejected_records = [record for record in records if record.get("rejected")]
    hallucination_records = [
        record
        for record in records
        if record.get("verifier_passed")
        and (
            not record.get("citations_present")
            or record.get("unknown_schema_usage")
            or record.get("quality_tag") in {"wrong_module_grounded", "sql_misuse"}
        )
    ]

    summary = {
        "label": label,
        "sample_size": len(records),
        "max_tokens": max_tokens,
        "passes": sum(1 for record in records if record.get("verifier_passed")),
        "rejections": sum(1 for record in records if record.get("rejected")),
        "rejection_rate_pct": round(sum(1 for record in records if record.get("rejected")) / max(len(records), 1) * 100, 2),
        "hallucination_rate_pct": round(len(hallucination_records) / max(len(records), 1) * 100, 2),
        "routing_accuracy_pct": round(
            sum(1 for record in records if quality_tag({**record, "quality_tag": record["quality_tag"]}) != "wrong_module_grounded")
            / max(len(records), 1)
            * 100,
            2,
        ),
        "doc_usage_pct": round(
            sum(1 for record in docs_records if record.get("docs_passed_to_prompt_count", 0) > 0)
            / max(len(docs_records), 1)
            * 100,
            2,
        ),
        "sql_correctness_pct": round(
            sum(1 for record in sql_records if record.get("verifier_passed") and not record.get("rejected"))
            / max(len(sql_records), 1)
            * 100,
            2,
        ),
        "sql_index_usage_pct": round(
            sum(1 for record in records if record.get("sql_index_used")) / max(len(records), 1) * 100,
            2,
        ),
        "citations_present_pct": round(
            sum(1 for record in records if record.get("citations_present")) / max(len(records), 1) * 100,
            2,
        ),
        "exact_module_top_hit_rate_pct": round(
            sum(1 for record in records if record.get("exact_module_top_hit")) / max(len(records), 1) * 100,
            2,
        ),
        "same_family_bleed_through_pct": round(
            sum(1 for record in records if record.get("same_family_bleed_through")) / max(len(records), 1) * 100,
            2,
        ),
        "task_match_rate_pct": round(
            sum(1 for record in docs_records if (record.get("task_semantic_strong_doc_count") or 0) > 0)
            / max(len(docs_records), 1)
            * 100,
            2,
        ),
        "weak_inference_rate_pct": round(
            sum(1 for record in records if record.get("quality_tag") == "weak_inference") / max(len(records), 1) * 100,
            2,
        ),
        "refusal_correctness_pct": round(
            sum(
                1
                for record in rejected_records
                if record.get("verifier_status") in {
                    "FAILED_TASK_SEMANTIC_NO_STRONG_MATCH",
                    "FAILED_TASK_MODULE_CORRECTION",
                    "FAILED_FINANCE_LEAF_NO_EXACT_DOCS",
                }
            )
            / max(len(rejected_records), 1)
            * 100,
            2,
        ),
        "quality_breakdown": dict(quality_counts),
        "top_failures": [record for record in records if record["quality_tag"] != "likely_correct"][:10],
        "good_examples": [record for record in records if record["quality_tag"] == "likely_correct"][:5],
    }
    return summary


async def run_validation(sample_size: int, max_tokens: int, label: str, bucket_plan: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    cases = focused_case_selection(bucket_plan or {}) if bucket_plan else balanced_case_selection(sample_size)
    tenant = SimpleNamespace(id="demo")
    engine = RAGEngine()
    results: List[Dict[str, Any]] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"real_model_validation_{label}.jsonl"
    summary_path = OUTPUT_DIR / f"real_model_validation_{label}_summary.json"

    for index, case in enumerate(cases, start=1):
        print(f"[real-model] case {index}/{len(cases)}: {case['id']} | {case_bucket(case)} | {case['question'][:90]}")
        request = ChatRequest(
            messages=[Message(role=Role.USER, content=case["question"])],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        started_at = time.perf_counter()

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
                "category": case.get("category"),
                "benchmark_module": case.get("module"),
                "benchmark_bucket": case_bucket(case),
                "difficulty": case.get("difficulty"),
                "question": case["question"],
                "benchmark_answer": case.get("answer"),
                "module_detected": response.audit.get("module"),
                "module_family_detected": response.audit.get("module_family"),
                "intent_detected": response.audit.get("task_type"),
                "retrieved_docs": [
                    {
                        "citation_id": chunk.get("citation_id"),
                        "title": chunk.get("title"),
                        "module": (chunk.get("metadata") or {}).get("module"),
                        "module_family": (chunk.get("metadata") or {}).get("module_family"),
                        "corpus": (chunk.get("metadata") or {}).get("corpus"),
                        "source_system": (chunk.get("metadata") or {}).get("source_system"),
                        "authority_tier": (chunk.get("metadata") or {}).get("authority_tier"),
                        "task_signals": (chunk.get("metadata") or {}).get("task_signals") or [],
                        "task_confidence": float((chunk.get("metadata") or {}).get("task_confidence") or 0.0),
                        "task_match_score": float((chunk.get("metadata") or {}).get("task_match_score") or 0.0),
                        "task_match_strength": (chunk.get("metadata") or {}).get("task_match_strength") or "none",
                        "score": float(chunk.get("combined_score") or chunk.get("score") or 0.0),
                        "source_uri": chunk.get("source_uri"),
                    }
                    for chunk in retrieved_chunks
                ],
                "retrieved_doc_count": len(retrieved_chunks),
                "docs_passed_to_prompt_count": int(response.audit.get("docs_passed_to_prompt_count") or 0),
                "schema_objects_used": schema_objects,
                "unknown_schema_usage": bool(
                    any(obj == "UNKNOWN" for obj in response.audit.get("normalization_tags", {}).get("unconfirmed", []))
                ),
                "sql_index_used": bool(response.audit.get("sql_index_used")),
                "sql_generated": extract_sql(output),
                "verifier_status": response.audit.get("verification_status"),
                "verifier_passed": response.audit.get("verification_status") == "PASSED",
                "citations_present": bool(response.citations),
                "citation_count": len(response.citations),
                "citations": [citation.model_dump() for citation in response.citations],
                "rejected": is_safe_rejection(output),
                "query_task_signals": response.audit.get("query_task_signals") or [],
                "task_semantic_gate": response.audit.get("task_semantic_gate"),
                "task_semantic_strong_doc_count": int(response.audit.get("task_semantic_strong_doc_count") or 0),
                "task_semantic_medium_doc_count": int(response.audit.get("task_semantic_medium_doc_count") or 0),
                "task_match_rate": float(response.audit.get("task_match_rate") or 0.0),
                "task_semantic_correction": response.audit.get("task_semantic_correction"),
                "response_time_sec": round(time.perf_counter() - started_at, 2),
                "output": output,
                "runtime_error": None,
            }
            record["exact_module_top_hit"] = exact_module_top_hit(record)
            record["same_family_bleed_through"] = same_family_bleed(record)
        except Exception as exc:
            record = {
                "id": case["id"],
                "category": case.get("category"),
                "benchmark_module": case.get("module"),
                "benchmark_bucket": case_bucket(case),
                "difficulty": case.get("difficulty"),
                "question": case["question"],
                "benchmark_answer": case.get("answer"),
                "module_detected": "UNKNOWN",
                "module_family_detected": "UNKNOWN",
                "intent_detected": None,
                "retrieved_docs": [],
                "retrieved_doc_count": 0,
                "docs_passed_to_prompt_count": 0,
                "schema_objects_used": [],
                "unknown_schema_usage": False,
                "sql_index_used": False,
                "sql_generated": None,
                "verifier_status": "RUNTIME_ERROR",
                "verifier_passed": False,
                "citations_present": False,
                "citation_count": 0,
                "citations": [],
                "rejected": True,
                "response_time_sec": round(time.perf_counter() - started_at, 2),
                "output": FAIL_CLOSED_MESSAGE,
                "runtime_error": str(exc),
                "exact_module_top_hit": False,
                "same_family_bleed_through": False,
            }

        record["quality_tag"] = quality_tag(record)
        results.append(record)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = summarize_records(results, label=label, max_tokens=max_tokens)
    summary["output_path"] = str(output_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run balanced real-model Oracle Fusion validation.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=140)
    parser.add_argument("--label", type=str, default="baseline")
    parser.add_argument("--bucket-plan", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_validation(
            sample_size=args.sample_size,
            max_tokens=args.max_tokens,
            label=args.label,
            bucket_plan=parse_bucket_plan(args.bucket_plan),
        )
    )
