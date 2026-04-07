import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import get_default_registry
from backend.core.grounding.verifier import Verifier
from backend.core.ingest.specialization_tracks import build_specialization_tracks
from backend.core.rag.engine import FAIL_CLOSED_MESSAGE, RAGEngine
from backend.core.schemas.api import ChatRequest, Message, Role

SPECIALIZATION_DIR = BASE_DIR / "specialization_tracks"
OUTPUT_DIR = SPECIALIZATION_DIR / "sql_rollout_readiness"
SUMMARY_PATH = SPECIALIZATION_DIR / "specialization_ingestion_summary.json"
SQL_PILOT_RESULTS_PATH = SPECIALIZATION_DIR / "pilot_benchmarks" / "sql_pilot_results.jsonl"
CERT_BENCH_RESULTS_PATH = BASE_DIR / "production_benchmark" / "production_1000_certification_v1" / "benchmark_results.jsonl"


def _load_summary(rebuild: bool) -> Dict[str, Any]:
    if rebuild or not SUMMARY_PATH.exists():
        return build_specialization_tracks()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _numeric_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", value or "")
    return (int(match.group(1)), value) if match else (0, value or "")


def _extract_sql(output: str) -> Optional[str]:
    match = re.search(r"\[SQL\](.*?)(?:\n\[[A-Z_]+\]|\Z)", output, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_sql_tables(sql: str, registry: Any) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="oracle")
    except Exception:
        return []
    tables = []
    for table in tree.find_all(exp.Table):
        canonical = registry.resolve_object_name(table.name.upper()) or table.name.upper()
        tables.append(canonical)
    return sorted(set(tables))


def _extract_sql_columns(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="oracle")
    except Exception:
        return []
    columns = []
    for column in tree.find_all(exp.Column):
        if column.name:
            columns.append(column.name.upper())
    return sorted(set(columns))


def _extract_sql_join_pairs(sql: str, registry: Any) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="oracle")
    except Exception:
        return []
    seen_tables: List[str] = []
    joins = set()
    for table in tree.find_all(exp.Table):
        canonical = registry.resolve_object_name(table.name.upper()) or table.name.upper()
        seen_tables.append(canonical)
    for idx in range(len(seen_tables) - 1):
        joins.add(f"{seen_tables[idx]}->{seen_tables[idx + 1]}")
    return sorted(joins)


def _case_difficulty(record: Dict[str, Any]) -> str:
    tables = len(record.get("tables_used", []))
    if tables >= 3:
        return "hard"
    if tables == 2:
        return "medium"
    return "easy"


def build_broad_sql_cases(summary: Dict[str, Any], total_cases: int = 200, refusal_cases: int = 40) -> List[Dict[str, Any]]:
    records = sorted(summary.get("sql_records", []), key=lambda row: (row["module"], _numeric_sort_key(row["title"])))
    by_module: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"join": [], "single": []})
    for record in records:
        bucket = "join" if len(record.get("tables_used", [])) > 1 else "single"
        by_module[record["module"]][bucket].append(record)

    modules = sorted(by_module)
    positive_target = max(total_cases - refusal_cases, 0)
    join_target = min(max(positive_target // 2, 40), sum(len(by_module[m]["join"]) for m in modules))
    selected: List[Dict[str, Any]] = []
    seen = set()
    positions = {module: {"join": 0, "single": 0} for module in modules}

    def take_round(bucket_name: str, target: int) -> None:
        while len([row for row in selected if row.get("_bucket") == bucket_name]) < target:
            progressed = False
            for module in modules:
                bucket = by_module[module][bucket_name]
                pos = positions[module][bucket_name]
                while pos < len(bucket) and bucket[pos]["content_hash"] in seen:
                    pos += 1
                positions[module][bucket_name] = pos
                if pos >= len(bucket):
                    continue
                record = dict(bucket[pos])
                record["_bucket"] = bucket_name
                selected.append(record)
                seen.add(record["content_hash"])
                positions[module][bucket_name] = pos + 1
                progressed = True
                if len([row for row in selected if row.get("_bucket") == bucket_name]) >= target:
                    break
            if not progressed:
                break

    take_round("join", join_target)
    take_round("single", max(positive_target - join_target, 0))

    if len(selected) < positive_target:
        for record in records:
            if record["content_hash"] in seen:
                continue
            selected.append(dict(record, _bucket="fill"))
            seen.add(record["content_hash"])
            if len(selected) >= positive_target:
                break

    cases: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected[:positive_target], start=1):
        tables_used = record.get("tables_used", [])
        difficulty = _case_difficulty(record)
        if record["task_type"] == "sql_troubleshooting":
            prompt = (
                f"Troubleshoot and return grounded Oracle SQL for Oracle Fusion {record['module']}: {record['title']}."
            )
        elif len(tables_used) > 1:
            prompt = (
                f"Generate grounded Oracle SQL for Oracle Fusion {record['module']} using validated joins for {record['title']}. "
                f"Ground the query on {', '.join(tables_used[:3])}."
            )
        else:
            prompt = f"Generate grounded Oracle SQL for Oracle Fusion {record['module']}: {record['title']}."
        cases.append(
            {
                "id": f"BSQL-{idx:03d}",
                "lane": "sql",
                "module": record["module"],
                "question": prompt,
                "expected_behavior": "sql",
                "task_type": record["task_type"],
                "difficulty": difficulty,
                "source_title": record["title"],
                "source_uri": record["source_uri"],
                "expected_tables": record["tables_used"][:12],
                "expected_columns": record["columns_used"][:24],
                "expected_joins": record.get("joins_used", [])[:12],
            }
        )

    negative_templates = [
        "Generate Oracle SQL for Oracle Fusion {module} using fake table XX_{prefix}_FAKE_HEADERS and SELECT *.",
        "Write Oracle SQL for Oracle Fusion {module} with TODO comments and placeholder table {prefix}_NOT_REAL.",
        "Troubleshoot this Oracle SQL for Oracle Fusion {module}: SELECT * FROM {prefix}_INVALID_TABLE.",
        "Generate Oracle SQL for Oracle Fusion {module} using DUAL filler and unknown table {prefix}_MISSING_DATA.",
    ]
    module_prefixes = {
        "Payables": "AP",
        "Receivables": "AR",
        "General Ledger": "GL",
        "Cash Management": "CE",
        "Assets": "FA",
        "Expenses": "EXM",
        "Procurement": "PO",
        "SCM": "INV",
        "HCM": "PER",
        "Projects": "PJF",
        "Tax": "ZX",
        "Financials": "FUN",
        "Manufacturing": "JMF",
        "Recruiting": "IRC",
    }
    negative_modules = modules or sorted(module_prefixes)
    neg_idx = 1
    while len(cases) < total_cases:
        for module in negative_modules:
            prefix = module_prefixes.get(module, re.sub(r"[^A-Z]", "", module.upper())[:3] or "XX")
            template = negative_templates[(neg_idx - 1) % len(negative_templates)]
            cases.append(
                {
                    "id": f"BSQL-N{neg_idx:03d}",
                    "lane": "sql",
                    "module": module,
                    "question": template.format(module=module, prefix=prefix),
                    "expected_behavior": "refusal",
                    "task_type": "sql_generation",
                    "difficulty": "medium",
                    "source_title": "negative_case",
                    "source_uri": "synthetic://negative/broad_sql",
                    "expected_tables": [],
                    "expected_columns": [],
                    "expected_joins": [],
                }
            )
            neg_idx += 1
            if len(cases) >= total_cases:
                break
    return cases[:total_cases]


def _score_sql_case(case: Dict[str, Any], record: Dict[str, Any], verifier: Verifier) -> Dict[str, Any]:
    registry = verifier.registry
    output = record["output"]
    sql_text = _extract_sql(output) or ""
    rejected = record["rejected"]

    if case["expected_behavior"] == "refusal":
        refusal_correct = rejected
        return {
            "refusal_correct": refusal_correct,
            "verifier_approved_sql": False,
            "semantic_match": False,
            "table_correctness": 1.0 if refusal_correct else 0.0,
            "column_correctness": 1.0 if refusal_correct else 0.0,
            "required_field_coverage": 1.0 if refusal_correct else 0.0,
            "join_correctness": 1.0 if refusal_correct else 0.0,
            "style_compliant": False,
            "module_correctness": 1.0 if refusal_correct else 0.0,
            "verifier_failure_reason": "" if refusal_correct else "under_refusal",
        }

    sql_ok, sql_reason = verifier.verify_sql(sql_text) if sql_text else (False, "missing_sql")
    style_ok, style_reason = verifier.verify_sql_style(sql_text) if sql_text else (False, "missing_sql")
    module_ok, module_reason = verifier.verify_module_alignment(sql_text, case["module"]) if sql_text else (False, "missing_sql")
    output_tables = set(_extract_sql_tables(sql_text, registry))
    output_columns = set(_extract_sql_columns(sql_text))
    output_joins = set(_extract_sql_join_pairs(sql_text, registry))
    expected_tables = set(case["expected_tables"])
    expected_columns = set(case["expected_columns"])
    table_correctness = len(output_tables & expected_tables) / max(len(expected_tables), 1)
    column_denominator = max(min(len(expected_columns), 12), 1)
    column_correctness = len(output_columns & expected_columns) / column_denominator
    expected_joins = {str(join).upper() for join in case.get("expected_joins", []) if str(join).strip()}
    if expected_joins:
        join_correctness = len({str(join).upper() for join in output_joins} & expected_joins) / max(len(expected_joins), 1)
    elif len(expected_tables) <= 1:
        join_correctness = 1.0
    else:
        join_correctness = 1.0 if len(output_tables) >= 2 else 0.0

    request_shape = {
        "required_fields": [
            {
                "label": column_name,
                "columns": [column_name],
                "aliases": [column_name],
            }
            for column_name in sorted(expected_columns)
        ],
        "required_entities": sorted(expected_tables),
        "requires_join": len(expected_tables) > 1,
        "required_filters": [],
    }
    shape_ok, shape_reason = verifier.verify_sql_request_shape(sql_text, request_shape) if sql_text else (False, "missing_sql")

    semantic_match = bool(
        case["source_title"] in " ".join(doc.get("title", "") for doc in record["retrieved_docs"])
        or case["source_uri"] in " ".join(doc.get("source_uri", "") for doc in record["retrieved_docs"])
    )
    verifier_reason = ""
    if not sql_ok:
        verifier_reason = sql_reason or ""
    elif not style_ok:
        verifier_reason = style_reason or ""
    elif not shape_ok:
        verifier_reason = shape_reason or ""

    return {
        "refusal_correct": False,
        "verifier_approved_sql": bool(sql_text) and sql_ok and style_ok and shape_ok and not rejected,
        "semantic_match": semantic_match,
        "table_correctness": round(min(table_correctness, 1.0), 4),
        "column_correctness": round(min(column_correctness, 1.0), 4),
        "required_field_coverage": round(min(column_correctness, 1.0), 4),
        "join_correctness": round(min(join_correctness, 1.0), 4),
        "style_compliant": bool(sql_text) and style_ok and not rejected,
        "module_correctness": 1.0 if (bool(sql_text) and module_ok and not rejected) else 0.0,
        "verifier_failure_reason": verifier_reason,
        "module_failure_reason": "" if module_ok else (module_reason or ""),
        "request_shape_ok": bool(sql_text) and shape_ok and not rejected,
        "output_tables": sorted(output_tables),
        "output_columns": sorted(output_columns),
        "output_joins": sorted(output_joins),
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _harvest_existing_sql_failures(registry: Any) -> Dict[str, Any]:
    pilot_rows = _read_jsonl(SQL_PILOT_RESULTS_PATH)
    cert_rows = _read_jsonl(CERT_BENCH_RESULTS_PATH)

    harvested: List[Dict[str, Any]] = []
    for row in pilot_rows:
        if row.get("verifier_approved_sql") or row.get("refusal_correct"):
            continue
        harvested.append(
            {
                "source": "sql_pilot",
                "case_id": row.get("id"),
                "module": row.get("module"),
                "tables_requested": row.get("expected_tables", []),
                "tables_used_in_output": row.get("output_tables", []),
                "join_pattern": row.get("output_joins", []),
                "verifier_failure_reason": row.get("verification_status") or row.get("verifier_failure_reason") or "",
                "semantic_mismatch_reason": "retrieved_source_miss" if not row.get("semantic_match") else "",
                "missing_metadata_flags": [],
            }
        )

    for row in cert_rows:
        if not (row.get("sql_index_used") or row.get("sql_generated")):
            continue
        if row.get("scoring_outcome") in {"grounded_correct", "safe_refusal_correct"}:
            continue
        flags = []
        for object_name in row.get("schema_objects_used") or []:
            entry = registry.get_entry(object_name)
            if not entry:
                flags.append(f"missing_object:{object_name}")
            elif str(entry.get("owning_module") or "") in {"UNKNOWN", "Common"}:
                flags.append(f"missing_ownership:{object_name}")
        harvested.append(
            {
                "source": "benchmark_1000",
                "case_id": row.get("id"),
                "module": row.get("benchmark_module"),
                "tables_requested": row.get("schema_objects_used", []),
                "tables_used_in_output": [],
                "join_pattern": [],
                "verifier_failure_reason": row.get("verifier_status") or "",
                "semantic_mismatch_reason": row.get("task_semantic_gate") or "",
                "missing_metadata_flags": sorted(set(flags)),
            }
        )

    return {
        "pilot_failures": sum(1 for row in harvested if row["source"] == "sql_pilot"),
        "benchmark_sql_related_failures": sum(1 for row in harvested if row["source"] == "benchmark_1000"),
        "rows": harvested,
    }


async def run_cases(cases: List[Dict[str, Any]], max_tokens: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    verifier = Verifier()
    engine = RAGEngine()
    tenant = SimpleNamespace(id="demo")
    results: List[Dict[str, Any]] = []
    output_path = OUTPUT_DIR / "broad_sql_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for idx, case in enumerate(cases, start=1):
        print(f"[broad-sql] case {idx}/{len(cases)}: {case['id']} | {case['module']} | {case['question'][:96]}")
        request = ChatRequest(
            messages=[Message(role=Role.USER, content=case["question"])],
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.12,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        response = await engine.chat(db=None, tenant=tenant, request=request, require_citations=True)
        output = response.choices[0]["message"]["content"]
        record = {
            **case,
            "module_detected": response.audit.get("module"),
            "intent_detected": response.audit.get("task_type"),
            "retrieved_docs": [
                {
                    "title": citation.title,
                    "module": citation.module,
                    "source_uri": citation.source_uri,
                }
                for citation in response.citations
            ],
            "verification_status": response.audit.get("verification_status"),
            "citations_present": bool(response.citations),
            "citation_count": len(response.citations),
            "rejected": FAIL_CLOSED_MESSAGE in output,
            "response_time_sec": round(time.perf_counter() - started, 2),
            "output": output,
            "audit": response.audit,
        }
        record.update(_score_sql_case(case, record, verifier))
        results.append(record)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return results, summarize_results(results)


def summarize_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    refusals = [record for record in records if record["expected_behavior"] == "refusal"]
    positives = [record for record in records if record["expected_behavior"] != "refusal"]
    failure_groups = Counter()
    by_module: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "positive_total": 0,
            "refusal_total": 0,
            "verifier_approved": 0,
            "semantic_match": 0,
            "module_correct": 0,
            "refusal_correct": 0,
            "style_compliant": 0,
            "join_correctness_sum": 0.0,
            "required_field_coverage_sum": 0.0,
        }
    )
    by_difficulty: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "positive_total": 0,
            "refusal_total": 0,
            "verifier_approved": 0,
            "semantic_match": 0,
            "refusal_correct": 0,
            "style_compliant": 0,
            "join_correctness_sum": 0.0,
            "required_field_coverage_sum": 0.0,
        }
    )

    for record in records:
        module_summary = by_module[record["module"]]
        module_summary["total"] += 1
        difficulty_summary = by_difficulty[record["difficulty"]]
        difficulty_summary["total"] += 1

        if record["expected_behavior"] == "refusal":
            module_summary["refusal_total"] += 1
            module_summary["refusal_correct"] += int(bool(record.get("refusal_correct")))
            difficulty_summary["refusal_total"] += 1
            difficulty_summary["refusal_correct"] += int(bool(record.get("refusal_correct")))
            if not record.get("refusal_correct"):
                failure_groups["over_refusal_or_under_refusal"] += 1
            continue

        module_summary["positive_total"] += 1
        module_summary["verifier_approved"] += int(bool(record.get("verifier_approved_sql")))
        module_summary["semantic_match"] += int(bool(record.get("semantic_match")))
        module_summary["module_correct"] += int(float(record.get("module_correctness", 0.0)) >= 1.0)
        module_summary["style_compliant"] += int(bool(record.get("style_compliant")))
        module_summary["join_correctness_sum"] += float(record.get("join_correctness", 0.0))
        module_summary["required_field_coverage_sum"] += float(record.get("required_field_coverage", 0.0))

        difficulty_summary["positive_total"] += 1
        difficulty_summary["verifier_approved"] += int(bool(record.get("verifier_approved_sql")))
        difficulty_summary["semantic_match"] += int(bool(record.get("semantic_match")))
        difficulty_summary["style_compliant"] += int(bool(record.get("style_compliant")))
        difficulty_summary["join_correctness_sum"] += float(record.get("join_correctness", 0.0))
        difficulty_summary["required_field_coverage_sum"] += float(record.get("required_field_coverage", 0.0))

        if not record.get("verifier_approved_sql"):
            reason = str(record.get("verifier_failure_reason") or "verifier_failure")
            if "not present in the Oracle Fusion metadata index" in reason:
                failure_groups["missing_table_or_view_ownership"] += 1
            elif "Join predicates are not grounded" in reason:
                failure_groups["incorrect_or_missing_join"] += 1
            elif "JOIN" in reason or "join" in reason:
                failure_groups["join_validation_failure"] += 1
            elif "comment" in reason.lower() or "placeholder" in reason.lower():
                failure_groups["output_formatting_issue"] += 1
            else:
                failure_groups["verifier_failure"] += 1
        elif not record.get("semantic_match"):
            failure_groups["semantic_mismatch"] += 1
        elif float(record.get("module_correctness", 0.0)) < 1.0:
            failure_groups["module_mismatch"] += 1

    summary = {
        "generated_at": time.time(),
        "total_cases": total,
        "positive_cases": len(positives),
        "refusal_cases": len(refusals),
        "verifier_approved_sql_pct": round(sum(1 for record in positives if record.get("verifier_approved_sql")) / max(len(positives), 1) * 100, 2),
        "semantic_match_pct": round(sum(1 for record in positives if record.get("semantic_match")) / max(len(positives), 1) * 100, 2),
        "avg_table_correctness_pct": round(sum(float(record.get("table_correctness", 0.0)) for record in positives) / max(len(positives), 1) * 100, 2),
        "avg_column_correctness_pct": round(sum(float(record.get("column_correctness", 0.0)) for record in positives) / max(len(positives), 1) * 100, 2),
        "avg_required_field_coverage_pct": round(sum(float(record.get("required_field_coverage", 0.0)) for record in positives) / max(len(positives), 1) * 100, 2),
        "join_correctness_pct": round(sum(float(record.get("join_correctness", 0.0)) for record in positives) / max(len(positives), 1) * 100, 2),
        "style_compliance_pct": round(sum(1 for record in positives if record.get("style_compliant")) / max(len(positives), 1) * 100, 2),
        "refusal_correctness_pct": round(sum(1 for record in refusals if record.get("refusal_correct")) / max(len(refusals), 1) * 100, 2),
        "module_correctness_pct": round(sum(float(record.get("module_correctness", 0.0)) for record in positives) / max(len(positives), 1) * 100, 2),
        "citation_presence_pct": round(sum(1 for record in records if record.get("citations_present")) / max(total, 1) * 100, 2),
        "no_hallucinated_schema_pct": round(sum(1 for record in positives if record.get("verifier_approved_sql")) / max(len(positives), 1) * 100, 2),
        "results_by_module": {
            module: {
                "total": data["total"],
                "positive_total": data["positive_total"],
                "refusal_total": data["refusal_total"],
                "verifier_approved_sql_pct": round(data["verifier_approved"] / max(data["positive_total"], 1) * 100, 2),
                "semantic_match_pct": round(data["semantic_match"] / max(data["positive_total"], 1) * 100, 2),
                "style_compliance_pct": round(data["style_compliant"] / max(data["positive_total"], 1) * 100, 2),
                "join_correctness_pct": round(data["join_correctness_sum"] / max(data["positive_total"], 1) * 100, 2),
                "required_field_coverage_pct": round(data["required_field_coverage_sum"] / max(data["positive_total"], 1) * 100, 2),
                "module_correctness_pct": round(data["module_correct"] / max(data["positive_total"], 1) * 100, 2),
                "refusal_correctness_pct": round(data["refusal_correct"] / max(data["refusal_total"], 1) * 100, 2) if data["refusal_total"] else None,
            }
            for module, data in sorted(by_module.items())
        },
        "results_by_difficulty": {
            difficulty: {
                "total": data["total"],
                "positive_total": data["positive_total"],
                "refusal_total": data["refusal_total"],
                "verifier_approved_sql_pct": round(data["verifier_approved"] / max(data["positive_total"], 1) * 100, 2),
                "semantic_match_pct": round(data["semantic_match"] / max(data["positive_total"], 1) * 100, 2),
                "style_compliance_pct": round(data["style_compliant"] / max(data["positive_total"], 1) * 100, 2),
                "join_correctness_pct": round(data["join_correctness_sum"] / max(data["positive_total"], 1) * 100, 2),
                "required_field_coverage_pct": round(data["required_field_coverage_sum"] / max(data["positive_total"], 1) * 100, 2),
                "refusal_correctness_pct": round(data["refusal_correct"] / max(data["refusal_total"], 1) * 100, 2) if data["refusal_total"] else None,
            }
            for difficulty, data in sorted(by_difficulty.items())
        },
        "top_failure_categories": dict(failure_groups.most_common(10)),
    }
    return summary


def build_failure_report(records: List[Dict[str, Any]], registry: Any) -> Dict[str, Any]:
    grouped = Counter()
    rows: List[Dict[str, Any]] = []
    for record in records:
        failed = False
        if record["expected_behavior"] == "refusal":
            failed = not record.get("refusal_correct")
        else:
            failed = not record.get("verifier_approved_sql") or not record.get("semantic_match") or float(record.get("module_correctness", 0.0)) < 1.0
        if not failed:
            continue

        missing_flags = []
        for object_name in record.get("expected_tables", []):
            entry = registry.get_entry(object_name)
            if not entry:
                missing_flags.append(f"missing_object:{object_name}")
            elif str(entry.get("owning_module") or "") in {"UNKNOWN", "Common"}:
                missing_flags.append(f"missing_ownership:{object_name}")
            if entry and str(entry.get("object_type") or "") == "view" and not entry.get("base_tables"):
                missing_flags.append(f"missing_view_lineage:{object_name}")
        expected_tables = record.get("expected_tables", [])
        if len(expected_tables) >= 2:
            for idx in range(len(expected_tables) - 1):
                if not registry.get_relation_details(expected_tables[idx], expected_tables[idx + 1]):
                    missing_flags.append(f"missing_relationship:{expected_tables[idx]}::{expected_tables[idx + 1]}")

        if record["expected_behavior"] == "refusal":
            failure_type = "over-refusal"
            if not record.get("rejected"):
                failure_type = "under-refusal"
        elif not record.get("verifier_approved_sql"):
            reason = str(record.get("verifier_failure_reason") or "")
            if any(flag.startswith("missing_object:") for flag in missing_flags):
                failure_type = "missing table ownership"
            elif any(flag.startswith("missing_view_lineage:") for flag in missing_flags):
                failure_type = "missing view ownership"
            elif any(flag.startswith("missing_relationship:") for flag in missing_flags):
                failure_type = "missing relationships (joins)"
            elif "Join predicates are not grounded" in reason:
                failure_type = "incorrect joins"
            elif "comment" in reason.lower() or "placeholder" in reason.lower():
                failure_type = "output formatting issues"
            else:
                failure_type = "semantic mismatch"
        elif not record.get("semantic_match"):
            failure_type = "semantic mismatch"
        elif float(record.get("module_correctness", 0.0)) < 1.0:
            failure_type = "semantic mismatch"
        else:
            failure_type = "semantic mismatch"

        grouped[failure_type] += 1
        rows.append(
            {
                "case_id": record["id"],
                "module": record["module"],
                "tables_requested": record.get("expected_tables", []),
                "tables_used_in_output": record.get("output_tables", []),
                "join_pattern": record.get("output_joins", []),
                "verifier_failure_reason": record.get("verifier_failure_reason", ""),
                "semantic_mismatch_reason": "" if record.get("semantic_match") else "retrieved_docs_missing_expected_source",
                "missing_metadata_flags": sorted(set(missing_flags)),
                "failure_group": failure_type,
            }
        )
    return {
        "total_failures": len(rows),
        "group_counts": dict(grouped.most_common()),
        "rows": rows,
    }


async def main_async(total_cases: int, refusal_cases: int, max_tokens: int, rebuild: bool) -> Dict[str, Any]:
    summary = _load_summary(rebuild)
    registry = get_default_registry()
    existing_failures = _harvest_existing_sql_failures(registry)
    cases = build_broad_sql_cases(summary, total_cases=total_cases, refusal_cases=refusal_cases)
    _write_jsonl(OUTPUT_DIR / "broad_sql_cases.jsonl", cases)
    _write_json(OUTPUT_DIR / "global_sql_failure_harvest.json", existing_failures)

    results, benchmark_summary = await run_cases(cases, max_tokens=max_tokens)
    failure_report = build_failure_report(results, registry)
    _write_jsonl(OUTPUT_DIR / "broad_sql_results.jsonl", results)
    _write_json(OUTPUT_DIR / "broad_sql_summary.json", benchmark_summary)
    _write_json(OUTPUT_DIR / "broad_sql_failure_report.json", failure_report)

    sql_manifest_rows = _read_jsonl(SPECIALIZATION_DIR / "manifests" / "sql_examples_manifest.jsonl")
    schema_manifest_rows = _read_jsonl(SPECIALIZATION_DIR / "manifests" / "schema_metadata_manifest.jsonl")
    schema_unknown = [
        name
        for name, entry in registry.objects.items()
        if str(entry.get("owning_module") or "") in {"UNKNOWN", "Common"}
        and any(name in (row.get("metadata", {}).get("tables_used") or []) for row in sql_manifest_rows)
    ]
    relation_pairs = len(registry.relation_details_by_pair)
    join_edges = sum(len(neighbors) for neighbors in registry.join_graph.values()) // 2

    readiness = {
        "total_sql_failures_fixed": max(existing_failures["pilot_failures"] + existing_failures["benchmark_sql_related_failures"] - failure_report["total_failures"], 0),
        "schema_coverage": {
            "sql_path_unknown_objects": len(schema_unknown),
            "registry_objects": len(registry.objects),
            "tables": sum(1 for entry in registry.objects.values() if entry.get("object_type") == "table"),
            "views": sum(1 for entry in registry.objects.values() if entry.get("object_type") == "view"),
        },
        "join_graph": {
            "relation_pairs": relation_pairs,
            "edges": join_edges,
        },
        "sql_pattern_count": len(sql_manifest_rows),
        "modules_still_weak": [
            module
            for module, data in benchmark_summary["results_by_module"].items()
            if (
                data["positive_total"] > 0
                and (data["verifier_approved_sql_pct"] < 85.0 or data["module_correctness_pct"] < 95.0)
            )
            or (
                data["refusal_total"] > 0
                and (data.get("refusal_correctness_pct") or 0.0) < 90.0
            )
        ],
        "go_decision": (
            "GO"
            if benchmark_summary["verifier_approved_sql_pct"] >= 85.0
            and benchmark_summary["semantic_match_pct"] >= 80.0
            and benchmark_summary["module_correctness_pct"] >= 95.0
            and benchmark_summary["refusal_correctness_pct"] >= 90.0
            and len(schema_unknown) == 0
            else "NO-GO"
        ),
    }
    _write_json(OUTPUT_DIR / "broad_sql_readiness.json", readiness)
    return {
        "benchmark_summary": benchmark_summary,
        "failure_report": failure_report,
        "readiness": readiness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--refusal-cases", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        main_async(
            total_cases=args.cases,
            refusal_cases=args.refusal_cases,
            max_tokens=args.max_tokens,
            rebuild=args.rebuild,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
